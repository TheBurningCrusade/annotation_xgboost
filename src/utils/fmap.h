#ifndef XGBOOST_UTILS_FMAP_H_
#define XGBOOST_UTILS_FMAP_H_
/*!
 * \file fmap.h
 * \brief helper class that holds the feature names and interpretations
 * \author Tianqi Chen
 */
#include <vector>
#include <string>
#include <cstring>
#include "./utils.h"

namespace xgboost {
namespace utils {
/*! \brief helper class that holds the feature names and interpretations 
 * 这里程序使用了libsvm输入数据格式的定义方式，需要对数据进行预处理，其中重要的
 * 步骤就是将每一维具体的特征名进行标号用整数代替，这个class的功能就是存储编号
 * 对应的特征名，文件的格式是特征标号/t特征名/t第三列，其中第三列代表什么还没有
 * 搞清楚, 但是他只能是i,q,int,float这四个字符或者字符串*/
class FeatMap {
 public:
  enum Type {
    kIndicator = 0,
    kQuantitive = 1,
    kInteger = 2,
    kFloat = 3
  };
  // function definitions
  /*! \brief load feature map from text format */
  inline void LoadText(const char *fname) {
    std::FILE *fi = utils::FopenCheck(fname, "r");
    this->LoadText(fi);
    std::fclose(fi);
  }
  /*! \brief load feature map from text format */
  inline void LoadText(std::FILE *fi) {
    int fid;
    char fname[1256], ftype[1256];
    while (std::fscanf(fi, "%d\t%[^\t]\t%s\n", &fid, fname, ftype) == 3) {
      this->PushBack(fid, fname, ftype);
    }
  }
  /*!\brief push back feature map */
  /* 没有将特征的编号存到里面，只是存了特征名和特征的的类型 */
  inline void PushBack(int fid, const char *fname, const char *ftype) {
    utils::Check(fid == static_cast<int>(names_.size()), "invalid fmap format");
    names_.push_back(std::string(fname));
    types_.push_back(GetType(ftype));
  }
  inline void Clear(void) {
    names_.clear(); types_.clear();
  }
  /*! \brief number of known features */
  size_t size(void) const {
    return names_.size();
  }
  /*! \brief return name of specific feature */
  const char* name(size_t idx) const {
    utils::Assert(idx < names_.size(), "utils::FMap::name feature index exceed bound");
    return names_[idx].c_str();
  }
  /*! \brief return type of specific feature */
  const Type& type(size_t idx) const {
    utils::Assert(idx < names_.size(), "utils::FMap::name feature index exceed bound");
    return types_[idx];
  }

 private:
  inline static Type GetType(const char *tname) {
    using namespace std;
    if (!strcmp("i", tname)) return kIndicator;
    if (!strcmp("q", tname)) return kQuantitive;
    if (!strcmp("int", tname)) return kInteger;
    if (!strcmp("float", tname)) return kFloat;
    utils::Error("unknown feature type, use i for indicator and q for quantity");
    return kIndicator;
  }
  // 对,私有类型下划线,记住了
  /*! \brief name of the feature */
  std::vector<std::string> names_;
  /*! \brief type of the feature */
  std::vector<Type> types_;
};

}  // namespace utils
}  // namespace xgboost
#endif  // XGBOOST_FMAP_H_
