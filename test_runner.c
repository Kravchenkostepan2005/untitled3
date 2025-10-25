/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#include <stdio.h>

int test_oid(void);
int test_ber(void);
int test_url(void);

int main(void) {
    int fails = 0;
    fails += test_oid();
    fails += test_ber();
    fails += test_url();
    if (fails == 0) {
        printf("OK\n");
        return 0;
    }
    printf("FAIL %d\n", fails);
    return 1;
}
