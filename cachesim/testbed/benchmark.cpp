// #include <string>
// #include <iostream>
#include <filesystem>
// #include <cstdio>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <malloc.h>

#include <ctime>
#include <chrono>

#define min(a,b) ((a)<(b)?(a):(b))
namespace fs = std::filesystem;

// ::time is the fastest for getting the second granularity steady clock
// through the vdso. This is faster than std::chrono::steady_clock::now and
// counting it as seconds since epoch.
inline uint32_t getCurrentTimeSec() {
  // time in seconds since epoch will fit in 32 bit. We use this primarily for
  // storing in cache.
  return static_cast<uint32_t>(std::time(nullptr));
}


int main(int argc, char const *argv[])
{
    std::string path = "/mnt/hdd/baleen/run";
    int i = 0;
    void *buffer = memalign(512, 8388608);
    uint32_t timeL = getCurrentTimeSec();
    int lastCnt = 0;
    for (const auto & entry : fs::directory_iterator(path)) {
        int iostart = 0;
        int ioend = 8388608;
        if (ioend > 8388608) {
          // XLOG(ERR) << "Exceed max length";
          ioend = 8388608;
        }
        // FILE *fin = fopen(, "rb");
        // ioend = 512*1024;
        // when it's larger than 512KB, it becomes multiple IOs?
        int fd = open(entry.path().string().c_str(), O_RDONLY | O_DIRECT); //O_RDWR O_RDONLY | 
        if (fd < 0) {
          // XLOG(ERR) << "File not found on HDD: " << entry;
            printf("File not found\n");
            return -1;
        } else {        
          // fseek(fin, iostart, SEEK_SET);
          int ret1 = lseek(fd, iostart, SEEK_SET);
          // for (int idx = iostart; idx < ioend && idx < 8388608; idx += 131072) {
          //   fread(&buffer, min(131072, ioend-idx), 1, fin);
          // }
          // fread(&buffer, ioend-iostart, 1, fin);
          int ret2 = read(fd, buffer, ioend-iostart);
          // min(8388608, ioend-iostart));
          close(fd);
          // fclose(fin);
          if (i % 1000 == 0) {
            printf("%d %d %d\n", fd, ret1, ret2);  
            printf("%d|%.*s|\n", i, 30, buffer);
          }
        }
        i += 1;
        if (getCurrentTimeSec() - timeL >= 1) {
          printf("%d %d\n", i-lastCnt, getCurrentTimeSec() - timeL);
          lastCnt = i;
          timeL = getCurrentTimeSec();
        }
    }
}
