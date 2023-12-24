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

// 5860533167
int main(int argc, char const *argv[])
{
    if (argc == 1) {
      printf("You need to supply a block device, e.g., %s /dev/sda\n", argv[0]);
      return -1;
    }
    const char* path = argv[1];
    int fd = open(path, O_RDONLY | O_DIRECT); // O_DIRECT
    if (fd == -1) {
      return -1;
    }

    long long ret1 = lseek(fd, 0, SEEK_END);
    printf("%lld\n", ret1);

    close(fd);
}
