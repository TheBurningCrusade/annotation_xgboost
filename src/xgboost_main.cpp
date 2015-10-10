#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#define NOMINMAX
#include <ctime>
#include <string>
#include <cstring>
#include "./sync/sync.h"
#include "io/io.h"
#include "utils/utils.h"
#include "utils/config.h"
#include "learner/learner-inl.hpp"

namespace xgboost {
/*!
 * \brief wrapping the training process 
 */
class BoostLearnTask {
 public:
  inline int Run(int argc, char *argv[]) {
    if (argc < 2) {
      printf("Usage: <config>\n");
      return 0;
    }    
    // 第一个参数是配置文件的路径，所以这里应该是读取配置文件中的信息，并进行了
    // 参数的初始化
    utils::ConfigIterator itr(argv[1]);
    while (itr.Next()) {
      this->SetParam(itr.name(), itr.val());
    }

    // 如果配置文件后还有参数，那么这里的for循环会进行参数的重置和覆盖
    for (int i = 2; i < argc; ++i) {
      char name[256], val[256];
      if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
        this->SetParam(name, val);
      }
    }

    // do not save anything when save to stdout
    if (model_out == "stdout" || name_pred == "stdout") {
      this->SetParam("silent", "1");
      save_period = 0;
    }
    // initialized the result
    rabit::Init(argc, argv);
    if (rabit::IsDistributed()) {
      std::string pname = rabit::GetProcessorName();
      fprintf(stderr, "start %s:%d\n", pname.c_str(), rabit::GetRank());
    }
    if (rabit::IsDistributed() && data_split == "NONE") {
      this->SetParam("dsplit", "row");
    }    
    if (rabit::GetRank() != 0) {
      this->SetParam("silent", "2");
    }    
    this->InitData();

    if (task == "train") {
      // if task is training, will try recover from checkpoint
      this->TaskTrain();
      return 0;
    } else {
      this->InitLearner();
    }
    if (task == "dump") {
      this->TaskDump(); return 0;
    }
    if (task == "eval") {
      this->TaskEval(); return 0;
    }
    if (task == "pred") {
      this->TaskPred();
    }
    return 0;
  }
  // 默认值name = "seed", val = "0"
  inline void SetParam(const char *name, const char *val) {
    // strcmp 相等返回0,加上!表示如果相等返回1，即执行语句
    if (!strcmp("silent", name)) silent = atoi(val);
    if (!strcmp("use_buffer", name)) use_buffer = atoi(val);
    if (!strcmp("num_round", name)) num_round = atoi(val);
    if (!strcmp("pred_margin", name)) pred_margin = atoi(val);
    if (!strcmp("ntree_limit", name)) ntree_limit = atoi(val);
    if (!strcmp("save_period", name)) save_period = atoi(val);
    if (!strcmp("eval_train", name)) eval_train = atoi(val);
    if (!strcmp("task", name)) task = val;
    if (!strcmp("data", name)) train_path = val;
    if (!strcmp("test:data", name)) test_path = val;
    if (!strcmp("model_in", name)) model_in = val;
    if (!strcmp("model_out", name)) model_out = val;
    if (!strcmp("model_dir", name)) model_dir_path = val;
    if (!strcmp("fmap", name)) name_fmap = val;
    if (!strcmp("name_dump", name)) name_dump = val;
    if (!strcmp("name_pred", name)) name_pred = val;
    if (!strcmp("dsplit", name)) data_split = val;
    if (!strcmp("dump_stats", name)) dump_model_stats = atoi(val);
    // strncmp 比较"eval[" 和name 的前5个字符,判断是否相等
    if (!strncmp("eval[", name, 5)) {
      char evname[256];
      // 从name中读取eval[之后的所有字符并切存入evname中,不读取右中括号],即 ^]
      /*
       scanf返回值为1表示读到了一个有效数据，sscanf（）的返回为读有有效数据的个数！
       例1：
       char str[]="124 abc" ;
       int a=0,b=0 ;
       sscanf( str , "%d %d" , &a , &b );  //想读入两个整数，但是串中只有一个
       数值123，另一个不是数值（abc）, 这时，读入的有效数据只有一个a=123, 而b
       没有读到有效数据，因此，返回值为1
       例2：
       char str[]="124 abc" ;
       char a[10] , b[10] ;
       sscanf( str , "%s %s" , a , b ); //这时会读到两个字符串，a="123" , b="abc"
       函数返回值 为2
       * */
      utils::Assert(sscanf(name, "eval[%[^]]", evname) == 1, "must specify evaluation name for display");
      eval_data_names.push_back(std::string(evname));
      eval_data_paths.push_back(std::string(val));
    }
    /* */ 
    learner.SetParam(name, val);
  }
 public:
  BoostLearnTask(void) {
    // default parameters
    silent = 0;
    use_buffer = 1;
    num_round = 10;
    save_period = 0;
    eval_train = 0;
    pred_margin = 0;
    ntree_limit = 0;
    dump_model_stats = 0;
    task = "train";
    model_in = "NULL";
    model_out = "NULL";
    name_fmap = "NULL";
    name_pred = "pred.txt";
    name_dump = "dump.txt";
    model_dir_path = "./";
    data_split = "NONE";
    load_part = 0;
    data = NULL;
  }
  ~BoostLearnTask(void){
    for (size_t i = 0; i < deval.size(); i++){
      delete deval[i];
    }
    if (data != NULL) delete data;
  }
 private:
  inline void InitData(void) {
    // 返回指向第一次出现字符character位置的指针，如果没找到则返回NULL, 
    // 如果找到返回的一个char × 类型
    if (strchr(train_path.c_str(), '%') != NULL) {
      char s_tmp[256];
      utils::SPrintf(s_tmp, sizeof(s_tmp), train_path.c_str(), rabit::GetRank());
      train_path = s_tmp;
      load_part = 1;
    }
    bool loadsplit = data_split == "row";
    if (name_fmap != "NULL") fmap.LoadText(name_fmap.c_str());
    if (task == "dump") return;
    if (task == "pred") {
      data = io::LoadDataMatrix(test_path.c_str(), silent != 0, use_buffer != 0, loadsplit);
    } else {
      // training
      /* 换回的是一个 DMatrixSimple类型的数据, 他有3个元素， 
       * 其中row_ptr_: 是CRS的索引数据 
       * row_data_:是稀疏元素的值， 他的每个元素包含两个值,一个是列的索引，
       * 另一个是元素的值
       * 第3个元素还没有看出他的意思 */
      data = io::LoadDataMatrix(train_path.c_str(),
                                silent != 0 && load_part == 0,
                                use_buffer != 0, loadsplit);
      utils::Assert(eval_data_names.size() == eval_data_paths.size(), "BUG");
      // 将训练过程中的验证数据也一并载入进去
      for (size_t i = 0; i < eval_data_names.size(); ++i) {
        deval.push_back(io::LoadDataMatrix(eval_data_paths[i].c_str(),
                                           silent != 0,
                                           use_buffer != 0,
                                           loadsplit));
        devalall.push_back(deval.back());
      }

      // dcache的第一个数据是训练数据，其他是训练过程中的验证数据
      std::vector<io::DataMatrix *> dcache(1, data);
      for (size_t i = 0; i < deval.size(); ++ i) {
        dcache.push_back(deval[i]);
      }
      // set cache data to be all training and evaluation data
      learner.SetCacheData(dcache);
      
      // add training set to evaluation set if needed
      if (eval_train != 0) {
        devalall.push_back(data);
        eval_data_names.push_back(std::string("train"));
      }
    }
  }
  inline void InitLearner(void) {
    if (model_in != "NULL") {
      learner.LoadModel(model_in.c_str());
    } else {
      utils::Assert(task == "train", "model_in not specified");
      learner.InitModel();
    }
  }
  inline void TaskTrain(void) {
    int version = rabit::LoadCheckPoint(&learner);
    if (version == 0) this->InitLearner();
    const time_t start = time(NULL);
    unsigned long elapsed = 0;
    learner.CheckInit(data);

    bool allow_lazy = learner.AllowLazyCheckPoint();
    // 这里就是表示要生成几个树，每一轮迭代都会生成一个固定深度树
    for (int i = version / 2; i < num_round; ++i) {
      elapsed = (unsigned long)(time(NULL) - start);
      if (version % 2 == 0) { 
        if (!silent) printf("boosting round %d, %lu sec elapsed\n", i, elapsed);
        // 负责每一轮建立树的模型，它包括所有的建立树模型的步骤
        learner.UpdateOneIter(i, *data);
        if (allow_lazy) {
          rabit::LazyCheckPoint(&learner);
        } else {
          rabit::CheckPoint(&learner);
        }
        // 如果vertion是奇数就执行上面的更新策略了,当然如果一开始version是偶数的话
        // 他会执行量词version+1，所以关键还是version的初始值,即它的初始值是奇数和偶数
        // 的处理不一样，开始为奇数不执行上面的语句,是偶数的话就执行
        version += 1;
      }
      utils::Assert(version == rabit::VersionNumber(), "consistent check");
      std::string res = learner.EvalOneIter(i, devalall, eval_data_names);
      if (rabit::IsDistributed()){
        if (rabit::GetRank() == 0) {
          rabit::TrackerPrintf("%s\n", res.c_str());
        }
      } else {
        if (silent < 2) {
          fprintf(stderr, "%s\n", res.c_str());
        }
      }
      if (save_period != 0 && (i + 1) % save_period == 0) {
        this->SaveModel(i);
      }
      if (allow_lazy) {
        rabit::LazyCheckPoint(&learner);
      } else {
        rabit::CheckPoint(&learner);
      }
      version += 1;
      utils::Assert(version == rabit::VersionNumber(), "consistent check");
      elapsed = (unsigned long)(time(NULL) - start);
    }
    // always save final round
    if ((save_period == 0 || num_round % save_period != 0) && model_out != "NONE") {
      if (model_out == "NULL"){
        this->SaveModel(num_round - 1);
      } else {
        this->SaveModel(model_out.c_str());
      }
    }
    if (!silent){
      printf("\nupdating end, %lu sec in all\n", elapsed);
    }
  }
  inline void TaskEval(void) {
    learner.EvalOneIter(0, devalall, eval_data_names);
  }
  inline void TaskDump(void){
    FILE *fo = utils::FopenCheck(name_dump.c_str(), "w");
    std::vector<std::string> dump = learner.DumpModel(fmap, dump_model_stats != 0);
    for (size_t i = 0; i < dump.size(); ++ i) {
      fprintf(fo,"booster[%lu]:\n", i);
      fprintf(fo,"%s", dump[i].c_str()); 
    }
    fclose(fo);
  }
  inline void SaveModel(const char *fname) const {
    if (rabit::GetRank() != 0) return;
    learner.SaveModel(fname);
  }
  inline void SaveModel(int i) const {
    char fname[256];
    sprintf(fname, "%s/%04d.model", model_dir_path.c_str(), i + 1);
    this->SaveModel(fname);
  }
  inline void TaskPred(void) {
    std::vector<float> preds;
    if (!silent) printf("start prediction...\n");
    learner.Predict(*data, pred_margin != 0, &preds, ntree_limit);
    if (!silent) printf("writing prediction to %s\n", name_pred.c_str());    
    FILE *fo;
    if (name_pred != "stdout") {
      fo = utils::FopenCheck(name_pred.c_str(), "w");
    } else {
      fo = stdout;
    }
    for (size_t i = 0; i < preds.size(); ++i) {
      fprintf(fo, "%g\n", preds[i]);
    }
    if (fo != stdout) fclose(fo);
  }
 private:
  /*! \brief whether silent */
  int silent;
  /*! \brief special load */
  int load_part;
  /*! \brief whether use auto binary buffer */
  int use_buffer;
  /*! \brief whether evaluate training statistics */            
  int eval_train;
  /*! \brief number of boosting iterations */
  int num_round;
  /*! \brief the period to save the model, 0 means only save the final round model */
  int save_period;
  /*! \brief the path of training/test data set */
  std::string train_path, test_path;
  /*! \brief the path of test model file, or file to restart training */
  std::string model_in;
  /*! \brief the path of final model file, to be saved */
  std::string model_out;
  /*! \brief the path of directory containing the saved models */
  std::string model_dir_path;
  /*! \brief task to perform */
  std::string task;
  /*! \brief name of predict file */
  std::string name_pred;
  /*! \brief data split mode */
  std::string data_split;
  /*!\brief limit number of trees in prediction */
  int ntree_limit;
  /*!\brief whether to directly output margin value */
  int pred_margin;
  /*! \brief whether dump statistics along with model */
  int dump_model_stats;
  /*! \brief name of feature map */
  std::string name_fmap;
  /*! \brief name of dump file */
  std::string name_dump;
  /*! \brief the paths of validation data sets */
  std::vector<std::string> eval_data_paths;
  /*! \brief the names of the evaluation data used in output log */
  std::vector<std::string> eval_data_names;
 private:
  // DataMatrix是个虚类，这里用指针有动态引用子类的作用
  // 训练数据
  io::DataMatrix* data;
  // 验证数据，可以有多个
  std::vector<io::DataMatrix*> deval;
  // 验证数据集，如果可能他的最后一个原始的训练数据 
  std::vector<const io::DataMatrix*> devalall;
  /* 用来存储特征编号，特征名，特征类型用 */
  utils::FeatMap fmap;
  learner::BoostLearner learner;
};
}

int main(int argc, char *argv[]){
  xgboost::BoostLearnTask tsk;
  tsk.SetParam("seed", "0");
  int ret = tsk.Run(argc, argv);
  rabit::Finalize();
  return ret;
}
