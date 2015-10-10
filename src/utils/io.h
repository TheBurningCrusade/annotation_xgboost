#ifndef XGBOOST_UTILS_IO_H
#define XGBOOST_UTILS_IO_H
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include "./utils.h"
#include "../sync/sync.h"
/*!
 * \file io.h
 * \brief general stream interface for serialization, I/O
 * \author Tianqi Chen
 */
namespace xgboost {
namespace utils {
// reuse the definitions of streams
typedef rabit::Stream IStream;
typedef rabit::utils::SeekStream ISeekStream;
typedef rabit::utils::MemoryFixSizeBuffer MemoryFixSizeBuffer;
typedef rabit::utils::MemoryBufferStream MemoryBufferStream;

/*! \brief implementation of file i/o stream */
class FileStream : public ISeekStream {
 public:
  explicit FileStream(std::FILE *fp) : fp(fp) {}
  explicit FileStream(void) {
    this->fp = NULL;
  }
  // fread(buffer,size,count,fp);
  // buffer：是一个指针，对fread来说，它是读入数据的存放地址。对fwrite来说，是要输出数据的地址。
  // size：要读写的字节数；
  // count:要进行读写多少个size字节的数据项；
  // fp:文件型指针。
  virtual size_t Read(void *ptr, size_t size) {
    return std::fread(ptr, size, 1, fp);
  }
  virtual void Write(const void *ptr, size_t size) {
    std::fwrite(ptr, size, 1, fp);
  }
  virtual void Seek(size_t pos) {
    std::fseek(fp, static_cast<long>(pos), SEEK_SET);
  }
  virtual size_t Tell(void) {
    return std::ftell(fp);
  }
  virtual bool AtEnd(void) const {
    return std::feof(fp) != 0;
  }
  inline void Close(void) {
    if (fp != NULL){
      std::fclose(fp); fp = NULL;
    }
  }

 private:
  std::FILE *fp;
};
}  // namespace utils
}  // namespace xgboost

#include "./base64-inl.h"
#endif
