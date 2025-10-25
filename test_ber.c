#include <stdio.h>
#include <string.h>
#include "ber.h"

int test_ber(void)
{
    uint8_t buf[64]; ber_buf_t b; ber_buf_init(&b, buf, sizeof(buf));
    if (ber_encode_integer(&b, 12345) != 0) { printf("encode int failed\n"); return 1; }
    size_t off = 0; uint8_t tag; const uint8_t *v; size_t vlen;
    if (ber_decode_tlv(buf, b.len, &off, &tag, &v, &vlen) != 0 || tag != 0x02) { printf("decode tlv failed\n"); return 1; }
    int64_t out; off = 0; if (ber_decode_integer(buf, b.len, &off, &out) != 0 || out != 12345) { printf("decode int value mismatch\n"); return 1; }
    return 0;
}
