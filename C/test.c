#include <stdio.h>
#include <sys/time.h>

int
main(int argc, char **argv)
{
    struct timeval time;
    struct timeval delay;
    int hh, mm, ss;
    int dsec;

    delay.tv_sec = 0;
    gettimeofday(&time, NULL);
    while (1) {
  /* Compute delay to next 0.1 second point */
  delay.tv_usec = 100000L - (time.tv_usec % 100000L);
  if (delay.tv_usec > 5000) {
      /* Wait until 5ms before then */
      delay.tv_usec -= 5000;
      select(0, NULL, NULL, NULL, &delay);
  }
  gettimeofday(&time, NULL);
  ss = (time.tv_sec % 86400); /* discard days */
  hh = ss /  3600;
  mm = (ss - hh*3600) / 60;
  ss %= 60;
  printf("\r%02d:%02d:%02d.%02d",
      hh, mm, ss, (time.tv_usec + 500) / 10000);
  fflush(stdout);
    }
}
