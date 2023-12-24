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
#include <random>
#include <iostream>
#include <ctime>
#include <chrono>

#define min(a,b) ((a)<(b)?(a):(b))

// ::time is the fastest for getting the second granularity steady clock
// through the vdso. This is faster than std::chrono::steady_clock::now and
// counting it as seconds since epoch.
inline uint32_t getCurrentTimeSec() {
  // time in seconds since epoch will fit in 32 bit. We use this primarily for
  // storing in cache.
  return static_cast<uint32_t>(std::time(nullptr));
}

inline uint64_t getCurrentTimeNs() {
  auto ret = std::chrono::steady_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(ret).count();
}

// 5860533167
int main(int argc, char const *argv[])
{
    if (argc == 1) {
      printf("You need to supply a block device, e.g., %s /dev/sda\n", argv[0]);
      return -1;
    }
    uint64_t low = 0, high = 1000*1000*1000*2900L;
    if (argc == 3) {
      switch(atoi(argv[2])) {
        // Orca - 4TB
        case 1: high = 1000*1000*1000*3900L; break;
        // Test first 10GB
        case 2: low = 0; high = 1000*1000*1000*10L; break;
        // Test middle 10GB
        case 3: low = 1000*1000*1000*2000L; high = 1000*1000*1000*2010L;
        // Narhwal - 3TB
        // default: break;
      }
    }
    const char* path = argv[1];
    // "/dev/sdc";
    printf("Starting in 3s with block device: %s %ld %ld\n", path, low, high);
    sleep(3);
    int i = 0;
    void *buffer = memalign(512, 8388608*4);
    uint64_t timeL = getCurrentTimeNs();
    int fd = open(path, O_RDONLY | O_DIRECT); // O_DIRECT
    std::uniform_int_distribution<uint64_t> uint_dist10(low,high);
    // printf("%d\n", fd);
    if (fd == -1) {
      return -1;
    }
    std::mt19937 gen(timeL);    
    uint64_t ioend = 8388608;
    if (ioend > 8388608) {
      // XLOG(ERR) << "Exceed max length";
      ioend = 8388608;
    }
    // FILE *fin = fopen(, "rb");
    // ioend = 512*2*1024*6;
    ioend = 512;
    // ioend = 524288;
    // int lengths[] = {
    //   // 1310720+1024,
    //   // 1310720+1024*16,
    //   // 1310720+1024*32,
    //   // 1310720+1024*48,
    //   // 1310720+1024*64,
    //   // 1310720+1024*32*3,
    //   // 1310720+1024*64*2,
    //   // 1310720+1024*32*5,
    //   // 1310720+1024*64*3,
    //   // 1310720+1024*32*7,
    //   // 1572864,
    //   // 1024*1024+(1+2+4+8+16+32+64+128)*1024,
    //   // 1024*1024+(2+4+8+16+32+64+128)*1024,
    //   // 1024*1024+(4+8+16+32+64+128)*1024,
    //   // 1024*1024+(8+16+32+64+128)*1024,
    //   // 1024*1024+(16+32+64+128)*1024,
    //   //                 1024*1024+(32+64+128)*1024,

    //                  // 1024*1024+256*1024,
    //                  // 1024*1024+(64+128)*1024,
    //                 // 1024*1024+512, 1024*1024+1024, 1024*1024+64*1024, 1024*1024+128*1024,
    //                  512, 1024, 2048, 4096,
    //                  8192, 16384, 32768, 65536,
    //                  128*1024, 256*1024, 384*1024, 512*1024, 5*128*1024, 768*1024, 7*128*1024, 1024*1024,
    //                 1024*1024+256*1024, 1024*1024+512*1024, 1024*1024+768*1024, 1024*1024*2, 1024*1024*3, 1024*1024*4, 1024*1024*5, 1024*1024*6, 1024*1024*7, 1024*1024*8};
    int lengths[2+8+64+7];
    lengths[0] = 512;
    lengths[1] = 16383*1024;
    for (int i = 0; i < 8; i++) {
      lengths[2+i] = (512 << i);
    }
    for (int i = 0; i < 64; i++) {
      lengths[2+8+i] = (i+1)*128*1024;
    }
    for (int i = 0; i < 7; i++) {
      lengths[2+8+64+i] = (i+1)*1024*1024;
    }

    for (auto bmlength : lengths) {
      int lastCnt = 0;
      int tt = 0;
      std::cerr << "# Start: " << bmlength << std::endl;
      uint64_t actualReadTime = 0;
      for (uint64_t i = 0; i < 10000; i++) {
          uint64_t idx = uint_dist10(gen);
          uint64_t iostart = 0;
          // when it's larger than 512KB, it becomes multiple IOs?
          // fseek(fin, iostart, SEEK_SET);
          uint64_t offset = idx - idx % 512 +iostart;
          // offset = 301334458993/1024*1024 + 262144;
          // ioend = 4587520;
          // iostart = 262144;
          ioend = bmlength;
          long long ret1 = lseek(fd, offset, SEEK_SET);
          if (ret1 < 0) {
            std::cerr << "Failed to seek to: " << offset << " " << ret1 << std::endl;
            return -1;
          }
          // for (int idx = iostart; idx < ioend && idx < 8388608; idx += 131072) {
          //   fread(&buffer, min(131072, ioend-idx), 1, fin);
          // }
          // fread(&buffer, ioend-iostart, 1, fin);
          uint64_t startTime = getCurrentTimeNs();
          int ret2 = read(fd, buffer, ioend-iostart);
          actualReadTime += getCurrentTimeNs() - startTime;
          if (actualReadTime < 10 * 1000000 || ret2 != bmlength) {
            // printf("//Short IO of %d %d: %lf\n", bmlength, ret2, actualReadTime/1e9);
          }
          // printf("%lld %d \n", ret1, ret2);
          // return -1;
          // min(8388608, ioend-iostart));
          // fclose(fin);
          if (i % 1000 == 0) {
            // printf("%d %lld %d %ld\n", fd, ret1, ret2, idx);  
            // ioend *= 2;
            // printf("%d|%.*s|\n", i, 30, buffer);
          }
          uint64_t timeWindow = 10 * 1000000000ULL;
          if (getCurrentTimeNs() - timeL >= timeWindow) {
            double actualTimePassed = (getCurrentTimeNs() - timeL) / 1000000000.0;
            // iops, latency, MB/s, bytes read
            if (tt > 0) {
              int iops = i-lastCnt;
              printf("%d, %lf, %lf, %d, %lf\n",
                     iops, actualTimePassed/iops, iops*ioend/1024.0/1024.0/actualTimePassed, bmlength, (double)actualReadTime/1e9/iops);
              // printf("// IOPS: %ld, latency: %lf, MB: %lf, Time Passed: %u, RandIdx: %lu, Offset: %lu, R1: %lld, R2: %d\n",
              //        i-lastCnt, 1.0/(i-lastCnt), (i-lastCnt)*ioend/1024.0/1024.0, getCurrentTimeSec() - timeL, idx, offset, ret1, ret2);

            }
            lastCnt = i;
            timeL = getCurrentTimeNs();
            actualReadTime = 0;
            tt++;
            if (tt >= 3) {
              break;
            }
          }
      }
    }
    close(fd);
}
