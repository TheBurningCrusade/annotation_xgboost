#ifndef XGBOOST_TREE_UPDATER_COLMAKER_INL_HPP_
#define XGBOOST_TREE_UPDATER_COLMAKER_INL_HPP_
/*!
 * \file updater_colmaker-inl.hpp
 * \brief use columnwise update to construct a tree
 * \author Tianqi Chen
 */
#include <vector>
#include <cmath>
#include <algorithm>
#include "./param.h"
#include "./updater.h"
#include "../utils/omp.h"
#include "../utils/random.h"
// #include <iostream>

namespace xgboost {
namespace tree {
/*! \brief colunwise update to construct a tree */
// 这个TStats的类型是GradStats，在当前目录的param.h中定义
template<typename TStats>
class ColMaker: public IUpdater {
 public:
  virtual ~ColMaker(void) {}
  // set training parameter
  virtual void SetParam(const char *name, const char *val) {
    param.SetParam(name, val);
  }
  virtual void Update(const std::vector<bst_gpair> &gpair,
                      IFMatrix *p_fmat,
                      const BoosterInfo &info,
                      const std::vector<RegTree*> &trees) {
    TStats::CheckInfo(info);
    // rescale learning rate according to size of trees
    float lr = param.learning_rate;
    param.learning_rate = lr / trees.size();
    // build tree
    // 对每个树启用builder进行更新, 注意这里trees的个数在普通情况下是=1的，只有
    // 随机森林的时候才会是别大于1的值
    for (size_t i = 0; i < trees.size(); ++i) {
      Builder builder(param);
      builder.Update(gpair, p_fmat, info, trees[i]);
    }

    param.learning_rate = lr;
  }

 protected:
  // training parameter
  TrainParam param;
  // data structure
  /*! \brief per thread x per node entry to store tmp data */
  struct ThreadEntry {
    /*! \brief statistics of data */
    // 这里的TStats的类型是GradStats，保存一个sum_grad和一个sum_hess
    TStats stats;
    /*! \brief extra statistics of data */
    TStats stats_extra;
    /*! \brief last feature value scanned */
    float  last_fvalue;
    /*! \brief first feature value scanned */
    float  first_fvalue;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit ThreadEntry(const TrainParam &param)
        : stats(param), stats_extra(param) {
    }
  };
  struct NodeEntry {
    /*! \brief statics for node entry */
    TStats stats;
    /*! \brief loss of this node, without split */
    bst_float root_gain;
    /*! \brief weight calculated related to current data */
    float weight;
    /*! \brief current best solution */
    SplitEntry best;
    // constructor
    explicit NodeEntry(const TrainParam &param)
        : stats(param), root_gain(0.0f), weight(0.0f){
    }
  };
  // actual builder that runs the algorithm
  struct Builder{
   public:
    // constructor
    explicit Builder(const TrainParam &param) : param(param) {}
    // update one tree, growing
    virtual void Update(const std::vector<bst_gpair> &gpair,
                        IFMatrix *p_fmat,
                        const BoosterInfo &info,
                        RegTree *p_tree) {
      // 下面这两行代码，感觉是出事话参数，然后将每个节点初始化的
      // 所有的g和h都加了起来，然后初始化了一个不对任何特征切分的一份
      // 数据，然后可能要用这份数据进行特征的遍历和切分，从而找到第一个
      // 根节点
      this->InitData(gpair, *p_fmat, info.root_index, *p_tree);
      this->InitNewNode(qexpand_, gpair, *p_fmat, info, *p_tree);
      // 按事先设置好的树的层次进行生成 
      for (int depth = 0; depth < param.max_depth; ++depth) {
        this->FindSplit(depth, qexpand_, gpair, p_fmat, info, p_tree);
        this->ResetPosition(qexpand_, p_fmat, *p_tree);
        this->UpdateQueueExpand(*p_tree, &qexpand_);
        this->InitNewNode(qexpand_, gpair, *p_fmat, info, *p_tree);
        // if nothing left to be expand, break
        if (qexpand_.size() == 0) break;
      }
      // set all the rest expanding nodes to leaf
      for (size_t i = 0; i < qexpand_.size(); ++i) {
        const int nid = qexpand_[i];
        (*p_tree)[nid].set_leaf(snode[nid].weight * param.learning_rate);
      }
      // remember auxiliary statistics in the tree node
      for (int nid = 0; nid < p_tree->param.num_nodes; ++nid) {
        p_tree->stat(nid).loss_chg = snode[nid].best.loss_chg;
        p_tree->stat(nid).base_weight = snode[nid].weight;
        p_tree->stat(nid).sum_hess = static_cast<float>(snode[nid].stats.sum_hess);
        snode[nid].stats.SetLeafVec(param, p_tree->leafvec(nid));
      }
    }

   protected:
    // initialize temp data structure
    inline void InitData(const std::vector<bst_gpair> &gpair,
                         const IFMatrix &fmat,
                         const std::vector<unsigned> &root_index, const RegTree &tree) {
      utils::Assert(tree.param.num_nodes == tree.param.num_roots, "ColMaker: can only grow new tree");
      const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
      {// setup position
        // 初步感觉positon存储的就是当前节点的父节点的索引下标
        position.resize(gpair.size());
        if (root_index.size() == 0) {
          for (size_t i = 0; i < rowset.size(); ++i) {
            position[rowset[i]] = 0;
          }
        } else {
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            position[ridx] = root_index[ridx];
            utils::Assert(root_index[ridx] < (unsigned)tree.param.num_roots, "root index exceed setting");
          }
        }
        // mark delete for the deleted datas
        for (size_t i = 0; i < rowset.size(); ++i) {
          const bst_uint ridx = rowset[i];
          if (gpair[ridx].hess < 0.0f) position[ridx] = ~position[ridx];
        }
        // mark subsample
        if (param.subsample < 1.0f) {
          for (size_t i = 0; i < rowset.size(); ++i) {
            const bst_uint ridx = rowset[i];
            // 和上一个for循环是一样的，但是这里对hess<0.0f的数据不做处理
            if (gpair[ridx].hess < 0.0f) continue;
            if (random::SampleBinary(param.subsample) == 0) position[ridx] = ~position[ridx];
          }
        }
      }
      {
        // initialize feature index
        // 将所有不为空的特征的特征编号保存到feat_index中
        unsigned ncol = static_cast<unsigned>(fmat.NumCol());
        for (unsigned i = 0; i < ncol; ++i) {
          if (fmat.GetColSize(i) != 0) {
            feat_index.push_back(i);
          }
        }
        unsigned n = static_cast<unsigned>(param.colsample_bytree * feat_index.size());
        random::Shuffle(feat_index);
        //utils::Check(n > 0, "colsample_bytree is too small that no feature can be included");
        utils::Check(n > 0, "colsample_bytree=%g is too small that no feature can be included", param.colsample_bytree);
        // feat_index 的前ncol个用来存储特征的索引下标,而后n-ncol个用来干什么？
        feat_index.resize(n);
      }
      {// setup temp space for each thread
       // std::vector< std::vector<ThreadEntry> > stemp;
        #pragma omp parallel
        {
          this->nthread = omp_get_num_threads();
        }
        // reserve a small space
        stemp.clear();
        // 这里对每一个线程都声明一个vector<ThreadEntry>，并且每个vector有256的预留空间
        stemp.resize(this->nthread, std::vector<ThreadEntry>());
        for (size_t i = 0; i < stemp.size(); ++i) {
          stemp[i].clear(); stemp[i].reserve(256);
        }
        snode.reserve(256);
      }
      {// expand query
        qexpand_.reserve(256); qexpand_.clear();
        // 注意:这里是num_roots，对已一个树模型这个变量是什么意思？
        // std::cout << "num_nodes::" << "    " << tree.param.num_nodes << std::endl;
        // std::cout << "num_roots::" << "    " << tree.param.num_roots << std::endl;
        // 好像这是tree.param.num_nodes=1也等于1
        // 这里tree.param.num_roots=1，不知为什么一个regtree结构为什么有num_roots这个变量，
        // regtree应该是代表一个树，怎么会出现有多个根的情况呢？
        for (int i = 0; i < tree.param.num_roots; ++i) {
          qexpand_.push_back(i);
        }
      }
    }
    /*! \brief initialize the base_weight, root_gain, and NodeEntry for all the new nodes in qexpand */
    // 英文已经的很明白了，这个函数的作用就是对于在qexpand中新生成的叶节点计算他们的在不分割情况下的目标
    // 损失函数的值，和每个叶节点的预测值即权重， 并将这些值都存储在snode中
    inline void InitNewNode(const std::vector<int> &qexpand,
                            const std::vector<bst_gpair> &gpair,
                            const IFMatrix &fmat,
                            const BoosterInfo &info,
                            const RegTree &tree) {
      {// setup statistics space for each tree node
        for (size_t i = 0; i < stemp.size(); ++i) {
          // 现在将每个线程下的vector<ThreadEntry>的大小都设置成了param.num_nodes的大小
          stemp[i].resize(tree.param.num_nodes, ThreadEntry(param));
        }
        // 同时snode的预留reserve空间的大小和每个线程下vector<ThreadEntry>的大小相同
        // 在这里也将snode的大小设成了和param.num_nodes的大小相同的值
        snode.resize(tree.param.num_nodes, NodeEntry(param));
        // std::cout << "num_nodes::" << "    " << tree.param.num_nodes << "roots: " << tree.param.num_roots << std::endl;
      }
      const std::vector<bst_uint> &rowset = fmat.buffered_rowset();
      // setup position
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        // 所有行数据的索引
        const bst_uint ridx = rowset[i];
        const int tid = omp_get_thread_num();
        if (position[ridx] < 0) continue;
        // 感觉postion代表的每个实例所属的叶节点的索引值
        // position就是每个实例在树中的父节点的索引,当然这里的父节点指的其实应该就是
        // 当前树的叶节点, 这里使用的omp每个实例的父节点的stats是分布在不同的线程的
        // 数据结构里面的
        stemp[tid][position[ridx]].stats.Add(gpair, info, ridx);
      }
      // sum the per thread statistics together
      for (size_t j = 0; j < qexpand.size(); ++j) {
        // nid 代表的应该是叶子节点, 即为即将扩展的节点
        const int nid = qexpand[j];
        TStats stats(param);
        // 这里就是将线程下的数据汇聚起来存到snode中
        for (size_t tid = 0; tid < stemp.size(); ++tid) {
          stats.Add(stemp[tid][nid].stats);
        }
        // update node statistics
        snode[nid].stats = stats;
        // 每个叶子节点的损失函数值, 所有叶子节点的root_gain加起来就是整个函数的目标函数
        snode[nid].root_gain = static_cast<float>(stats.CalcGain(param));
        // 每个叶子节点的权重,这里权重就是值的每个叶节点的预测值
        snode[nid].weight = static_cast<float>(stats.CalcWeight(param));
      }
    }
    /*! \brief update queue expand add in new leaves */
    inline void UpdateQueueExpand(const RegTree &tree, std::vector<int> *p_qexpand) {
      std::vector<int> &qexpand = *p_qexpand;
      std::vector<int> newnodes;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[ nid ].is_leaf()) {
          newnodes.push_back(tree[nid].cleft());
          newnodes.push_back(tree[nid].cright());
        }
      }
      // use new nodes for qexpand
      qexpand = newnodes;
    }    
    // parallel find the best split of current fid
    // this function does not support nested functions
    inline void ParallelFindSplit(const ColBatch::Inst &col,
                                  bst_uint fid,
                                  const IFMatrix &fmat,
                                  const std::vector<bst_gpair> &gpair,
                                  const BoosterInfo &info) {
      bool need_forward = param.need_forward_search(fmat.GetColDensity(fid));
      bool need_backward = param.need_backward_search(fmat.GetColDensity(fid));
      const std::vector<int> &qexpand = qexpand_;
      #pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp[tid];
        // cleanup temp statistics
        for (size_t j = 0; j < qexpand.size(); ++j) {
          temp[qexpand[j]].stats.Clear();
        }
        nthread = omp_get_num_threads();
        bst_uint step = (col.length + nthread - 1) / nthread;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position[ridx];
          if (nid < 0) continue;
          const float fvalue = col[i].fvalue;
          if (temp[nid].stats.Empty()) {
            temp[nid].first_fvalue = fvalue;
          }
          temp[nid].stats.Add(gpair, info, ridx);
          temp[nid].last_fvalue = fvalue;
        }
      }
      // start collecting the partial sum statistics
      bst_omp_uint nnode = static_cast<bst_omp_uint>(qexpand.size());
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint j = 0; j < nnode; ++j) {
        const int nid = qexpand[j];
        TStats sum(param), tmp(param), c(param);
        for (int tid = 0; tid < nthread; ++tid) {
          tmp = stemp[tid][nid].stats;
          stemp[tid][nid].stats = sum;
          sum.Add(tmp);
          if (tid != 0) {
            std::swap(stemp[tid - 1][nid].last_fvalue, stemp[tid][nid].first_fvalue);
          }
        }
        for (int tid = 0; tid < nthread; ++tid) {
          stemp[tid][nid].stats_extra = sum;
          ThreadEntry &e = stemp[tid][nid];
          float fsplit;
          if (tid != 0) {
            if(std::abs(stemp[tid - 1][nid].last_fvalue - e.first_fvalue) > rt_2eps) {
              fsplit = (stemp[tid - 1][nid].last_fvalue - e.first_fvalue) * 0.5f;
            } else {
              continue;
            }
          } else {
            fsplit = e.first_fvalue - rt_eps;
          }                        
          if (need_forward && tid != 0) {
            c.SetSubstract(snode[nid].stats, e.stats);
            if (c.sum_hess >= param.min_child_weight && e.stats.sum_hess >= param.min_child_weight) {
              bst_float loss_chg = static_cast<bst_float>(e.stats.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, false);
            }
          }
          if (need_backward) {
            tmp.SetSubstract(sum, e.stats);
            c.SetSubstract(snode[nid].stats, tmp);
            if (c.sum_hess >= param.min_child_weight && tmp.sum_hess >= param.min_child_weight) {
              bst_float loss_chg = static_cast<bst_float>(tmp.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
              e.best.Update(loss_chg, fid, fsplit, true);
            }
          }
        }
        if (need_backward) {
          tmp = sum;
          ThreadEntry &e = stemp[nthread-1][nid];
          c.SetSubstract(snode[nid].stats, tmp);
          if (c.sum_hess >= param.min_child_weight && tmp.sum_hess >= param.min_child_weight) {
            bst_float loss_chg = static_cast<bst_float>(tmp.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
            e.best.Update(loss_chg, fid, e.last_fvalue + rt_eps, true);
          }
        }
      }
      // rescan, generate candidate split
      #pragma omp parallel
      {
        TStats c(param), cright(param);
        const int tid = omp_get_thread_num();
        std::vector<ThreadEntry> &temp = stemp[tid];
        nthread = static_cast<bst_uint>(omp_get_num_threads());
        bst_uint step = (col.length + nthread - 1) / nthread;
        bst_uint end = std::min(col.length, step * (tid + 1));
        for (bst_uint i = tid * step; i < end; ++i) {
          const bst_uint ridx = col[i].index;
          const int nid = position[ridx];
          if (nid < 0) continue;
          const float fvalue = col[i].fvalue;
          // get the statistics of nid
          ThreadEntry &e = temp[nid];
          if (e.stats.Empty()) {
            e.stats.Add(gpair, info, ridx);
            e.first_fvalue = fvalue;
          } else {
            // forward default right
            if (std::abs(fvalue - e.first_fvalue) > rt_2eps){
              if (need_forward) { 
                c.SetSubstract(snode[nid].stats, e.stats);
                if (c.sum_hess >= param.min_child_weight && e.stats.sum_hess >= param.min_child_weight) {
                  bst_float loss_chg = static_cast<bst_float>(e.stats.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
                  e.best.Update(loss_chg, fid, (fvalue + e.first_fvalue) * 0.5f, false);
                }
              }
              if (need_backward) {
                cright.SetSubstract(e.stats_extra, e.stats);
                c.SetSubstract(snode[nid].stats, cright);
                if (c.sum_hess >= param.min_child_weight && cright.sum_hess >= param.min_child_weight) {
                  bst_float loss_chg = static_cast<bst_float>(cright.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
                  e.best.Update(loss_chg, fid, (fvalue + e.first_fvalue) * 0.5f, true);
                }
              }
            }          
            e.stats.Add(gpair, info, ridx);
            e.first_fvalue = fvalue;            
          }
        }
      }
    }    
    // enumerate the split values of specific feature
    inline void EnumerateSplit(const ColBatch::Entry *begin,
                               const ColBatch::Entry *end,
                               int d_step,
                               bst_uint fid,
                               const std::vector<bst_gpair> &gpair,
                               const BoosterInfo &info,
                               std::vector<ThreadEntry> &temp) {
      const std::vector<int> &qexpand = qexpand_;
      // clear all the temp statistics
      // temp表示当前线程下的数据结构,一个线程会分一个temp下的vector，这里是先清空
      // 需要扩展的节点的stats,需要扩展的节点都是叶节点
      for (size_t j = 0; j < qexpand.size(); ++j) {
        temp[qexpand[j]].stats.Clear();
      }
      // left statistics
      TStats c(param);
      // 这里我们取的是一列特征的特征值,这个for循环就是这列特征的一个特征值
      // 对于这些特征值我们就更新他们的父节点的temp[nid]
      // 这里我们必须注意特征值是有序的，即它是事先排序好的，每一个分割都是一个二叉树性质的东西
      for(const ColBatch::Entry *it = begin; it != end; it += d_step) {
        // 该特征的索引||  不对，他不是特征的索引，他标识的是Entry中的index，这个
        // index表示的就是CSC矩阵的index,如果以列为存储目标的话,那就是index表示的
        // 就是行号了,当然如果以行为存储目标的话，那就是列号了
        // 上面的理解应该不太对， ridx应该是该特征值所在实例所属于的叶节点
        const bst_uint ridx = it->index;
        // 该特征值的父节点的特征的索引，每个特征下会有很多的特征值，而这些特征值
        // 的postion是不一样的，确认一下这个position的真正意义是记录的它的父节点吗?
        // 如果是的话，这就表明一维特征是可以多次出现在一个树中的
        const int nid = position[ridx];
        // std::cout << "ridx: " << ridx << std::endl;
        // std::cout << "nid: " << nid << std::endl;
        if (nid < 0) continue;
        // start working
        const float fvalue = it->fvalue;
        // get the statistics of nid
        // 父节点的信息, 注意这里e只是一个引用
        ThreadEntry &e = temp[nid];
        // test if first hit, this is fine, because we set 0 during init
        if (e.stats.Empty()) {
          // 这里更新了temp[nid]是用来下面的for循环的
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        } else {
          // try to find a split
          if (std::abs(fvalue - e.last_fvalue) > rt_2eps && e.stats.sum_hess >= param.min_child_weight) {
            // 用没有分割过的父节点和分割节点计算
            // 这里是使用它们的相减值直接对c进行赋值，会覆盖c之前的值
            c.SetSubstract(snode[nid].stats, e.stats);
            if (c.sum_hess >= param.min_child_weight) {
              // snode[nid] 为父节点的信息
              // 这里相减的顺序和文章写的是相反的,注意一下, 这里在更新切分的索引是更新的是该特征的第一个特征值的索引，而不是
              // 该特征的索引
              bst_float loss_chg = static_cast<bst_float>(e.stats.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
              e.best.Update(loss_chg, fid, (fvalue + e.last_fvalue) * 0.5f, d_step == -1);
            }
          }
          // update the statistics
          // 如果特征值不有序的话，那么不能对他们的stats进行累加
          e.stats.Add(gpair, info, ridx);
          e.last_fvalue = fvalue;
        }
      }
      // std::cout << "dddd" << std::endl;
      // finish updating all statistics, check if it is possible to include all sum statistics
      // 在同一维特征下，不同的特征值都对他们的temp[nid]进行了更新，对于更新之后的temp[nid]我们要
      // 看看它是否是将要扩展的点(即当前是叶子节点，但这次要往下扩展一层，层位节点）的nid相等，如果
      // 相等就看是不是可以成为最佳切分点
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        ThreadEntry &e = temp[nid];
        c.SetSubstract(snode[nid].stats, e.stats);
        if (e.stats.sum_hess >= param.min_child_weight && c.sum_hess >= param.min_child_weight) {
          bst_float loss_chg = static_cast<bst_float>(e.stats.CalcGain(param) + c.CalcGain(param) - snode[nid].root_gain);
          const float gap = std::abs(e.last_fvalue) + rt_eps;
          const float delta = d_step == +1 ? gap: -gap;
          e.best.Update(loss_chg, fid, e.last_fvalue + delta, d_step == -1);
        }
      }
    }
    // update the solution candidate 
    virtual void UpdateSolution(const ColBatch &batch,
                                const std::vector<bst_gpair> &gpair,
                                const IFMatrix &fmat,
                                const BoosterInfo &info) {
      // start enumeration
      // 这个矩阵一共有多少列?
      const bst_omp_uint nsize = static_cast<bst_omp_uint>(batch.size);
      #if defined(_OPENMP)                                                                
      const int batch_size = std::max(static_cast<int>(nsize / this->nthread / 32), 1);
      #endif
      int poption = param.parallel_option;
      if (poption == 2) {
        poption = static_cast<int>(nsize) * 2 < nthread ? 1 : 0;
      }
      if (poption == 0) {
        // 一直都是0  poption 在demo的binary clacify中
        // std::cout << "poption: " << poption << "ddd" << std::endl;
        #pragma omp parallel for schedule(dynamic, batch_size)
        // 这里是将每一维特征分给不同的线程进行处理
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          // 这里是对所有遍历所有的特征，对于每一个特征下的特征值所在的实例都会属于
          // 不同的叶节点，所以每一个特征下的特征值要对所有的叶节点进行分割的，即表示
          // 一个叶节点下的所有分割必须在所有的特征下进行更新，而一个特征下的所有特征值
          // 的遍历可以为所有叶节点找到在该特征下的最优分割
          // 第i列中第一个特征开始的地方，fid
          const bst_uint fid = batch.col_index[i];
          const int tid = omp_get_thread_num();
          // 取到这一列特征
          const ColBatch::Inst c = batch[i];
          // GetColDensity的功能是计算这列这种中，非空的比例
          // 为什么需要从两个方向进行遍历？
          if (param.need_forward_search(fmat.GetColDensity(fid))) {
            // c.data 是一个指针const Entry *data
            this->EnumerateSplit(c.data, c.data + c.length, +1, 
                fid, gpair, info, stemp[tid]);
          }
          if (param.need_backward_search(fmat.GetColDensity(fid))) {
            this->EnumerateSplit(c.data + c.length - 1, c.data - 1, -1, 
                                 fid, gpair, info, stemp[tid]);
          }
        }
      } else {
        for (bst_omp_uint i = 0; i < nsize; ++i) {
          this->ParallelFindSplit(batch[i], batch.col_index[i],
                                  fmat, gpair, info);
        }
      } 
    }
    // find splits at current level, do split per level
    // 这个函数针对的单位是一个数据矩阵，然后是树的一层
    inline void FindSplit(int depth,
                          const std::vector<int> &qexpand,
                          const std::vector<bst_gpair> &gpair,
                          IFMatrix *p_fmat,
                          const BoosterInfo &info,
                          RegTree *p_tree) {
      std::vector<bst_uint> feat_set = feat_index;
      if (param.colsample_bylevel != 1.0f) {
        random::Shuffle(feat_set);
        unsigned n = static_cast<unsigned>(param.colsample_bylevel * feat_index.size());
        utils::Check(n > 0, "colsample_bylevel is too small that no feature can be included");
        feat_set.resize(n);
      }
      // 存储多组数据的迭代器，每个代表一批数据
      utils::IIterator<ColBatch> *iter = p_fmat->ColIterator(feat_set);
      while (iter->Next()) {
        this->UpdateSolution(iter->Value(), gpair, *p_fmat, info);
      }
      // after this each thread's stemp will get the best candidates, aggregate results
      this->SyncBestSolution(qexpand);
      // get the best result, we can synchronize the solution
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        NodeEntry &e = snode[nid];        
        // now we know the solution in snode[nid], set split
        if (e.best.loss_chg > rt_eps) {
          p_tree->AddChilds(nid);
          (*p_tree)[nid].set_split(e.best.split_index(), e.best.split_value, e.best.default_left());
          // mark right child as 0, to indicate fresh leaf
          (*p_tree)[(*p_tree)[nid].cleft()].set_leaf(0.0f, 0);
          (*p_tree)[(*p_tree)[nid].cright()].set_leaf(0.0f, 0);
        } else {
          (*p_tree)[nid].set_leaf(e.weight * param.learning_rate);
        }
      } 
    }
    // reset position of each data points after split is created in the tree
    inline void ResetPosition(const std::vector<int> &qexpand, IFMatrix *p_fmat, const RegTree &tree) {
      // set the positions in the nondefault
      this->SetNonDefaultPosition(qexpand, p_fmat, tree);      
      // set rest of instances to default position
      const std::vector<bst_uint> &rowset = p_fmat->buffered_rowset();
      // set default direct nodes to default
      // for leaf nodes that are not fresh, mark then to ~nid, 
      // so that they are ignored in future statistics collection
      const bst_omp_uint ndata = static_cast<bst_omp_uint>(rowset.size());
      
      #pragma omp parallel for schedule(static)
      for (bst_omp_uint i = 0; i < ndata; ++i) {
        const bst_uint ridx = rowset[i];
        if (ridx >= position.size()) {
          utils::Printf("ridx exceed bound\n");
        }
        const int nid = this->DecodePosition(ridx);
        if (tree[nid].is_leaf()) {
          // mark finish when it is not a fresh leaf
          if (tree[nid].cright() == -1) {
            position[ridx] = ~nid;
          }
        } else {
          // push to default branch
          // 应该是用来处理在该分割特征上没有特征值的实例?
          if (tree[nid].default_left()) {
            this->SetEncodePosition(ridx, tree[nid].cleft());
          } else {
            this->SetEncodePosition(ridx, tree[nid].cright());
          }
        }
      }
    }
    // customization part
    // synchronize the best solution of each node
    // 一个线程可能要承担多个特征的更新，而一个线程下只会留下一个特征的最优分割，
    // 这里是遍历所有线程先的最优分割然后找到一个叶节点下的最优分割
    virtual void SyncBestSolution(const std::vector<int> &qexpand) {
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        NodeEntry &e = snode[nid];
        // 这里表明需要遍历所有tid, 都有nid
        for (int tid = 0; tid < this->nthread; ++tid) {
          e.best.Update(stemp[tid][nid].best);
        }
      }
    }
    virtual void SetNonDefaultPosition(const std::vector<int> &qexpand,
                                       IFMatrix *p_fmat, const RegTree &tree) {
      // step 1, classify the non-default data into right places
      std::vector<unsigned> fsplits;
      for (size_t i = 0; i < qexpand.size(); ++i) {
        const int nid = qexpand[i];
        if (!tree[nid].is_leaf()) {
          // 如果该节点不是叶节点，就将他的分割index取出来
          fsplits.push_back(tree[nid].split_index());
        }
      }
      std::sort(fsplits.begin(), fsplits.end());
      fsplits.resize(std::unique(fsplits.begin(), fsplits.end()) - fsplits.begin());

      utils::IIterator<ColBatch> *iter = p_fmat->ColIterator(fsplits);
      while (iter->Next()) {
        const ColBatch &batch = iter->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          // 取一列数据
          ColBatch::Inst col = batch[i];
          // 取该列数据的第一个值的索引,即分割索引
          const bst_uint fid = batch.col_index[i];
          // 该列数据的长度,该特征下特征值的个数
          const bst_omp_uint ndata = static_cast<bst_omp_uint>(col.length);
          #pragma omp parallel for schedule(static)
          // 我们遍历该特征下所有的特征值,找到该特征当前的postion节点
          // 其实的值都被包括在qexpand中，只是qexpand中还会有不属于他的值，
          // 即一个本身的叶节点却没有扩展成功
          for (bst_omp_uint j = 0; j < ndata; ++j) {
            const bst_uint ridx = col[j].index;
            const float fvalue = col[j].fvalue;
            const int nid = this->DecodePosition(ridx);
            // go back to parent, correct those who are not default
            if (!tree[nid].is_leaf() && tree[nid].split_index() == fid) {
              // 扩展之前这些nid都表示叶节点,扩展后它不在是也节点，那么原来
              // 属于它的实例集合要根据它的切割阀值被分到它的左叶节点和右叶节点
              if(fvalue < tree[nid].split_cond()) {
                this->SetEncodePosition(ridx, tree[nid].cleft());
              } else {
                this->SetEncodePosition(ridx, tree[nid].cright());
              }
            }
          }
        }
      }
    }
    // utils to get/set position, with encoded format
    // return decoded position
    inline int DecodePosition(bst_uint ridx) const{
      const int pid = position[ridx];
      return pid < 0 ? ~pid : pid;
    }
    // encode the encoded position value for ridx
    inline void SetEncodePosition(bst_uint ridx, int nid) {
      if (position[ridx] < 0) {
        position[ridx] = ~nid;
      } else {
        position[ridx] = nid;
      }
    }
    //--data fields--
    const TrainParam &param;
    // number of omp thread used during training
    int nthread;
    // Per feature: shuffle index of each feature index
    std::vector<bst_uint> feat_index;
    // Instance Data: current node position in the tree of each instance
    std::vector<int> position;
    // PerThread x PerTreeNode: statistics for per thread construction
    // stemp的第一层是线程的个数
    // 第二层节点的个数
    std::vector< std::vector<ThreadEntry> > stemp;
    /*! \brief TreeNode Data: statistics for each constructed node */
    std::vector<NodeEntry> snode;
    /*! \brief queue of nodes to be expanded */
    std::vector<int> qexpand_;
  };
};

}  // namespace tree
}  // namespace xgboost
#endif  // XGBOOST_TREE_UPDATER_COLMAKER_INL_HPP_
