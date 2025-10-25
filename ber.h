#ifndef BER_H
#define BER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} ber_buf_t;

void ber_buf_init(ber_buf_t *b, uint8_t *storage, size_t cap);
size_t ber_buf_write(ber_buf_t *b, const void *src, size_t n);

// TLV encoding
int ber_encode_tlv(ber_buf_t *b, uint8_t tag, const uint8_t *value, size_t vlen);
int ber_encode_integer(ber_buf_t *b, int64_t value);
int ber_encode_octet_string(ber_buf_t *b, const uint8_t *str, size_t slen);
int ber_encode_oid(ber_buf_t *b, const uint32_t *oid, size_t oid_len);

// Constructed types helpers
int ber_start_sequence(ber_buf_t *b, size_t *len_pos);
int ber_end_sequence(ber_buf_t *b, size_t len_pos);

// New: generic constructed with arbitrary tag (e.g. 0xA0 for GetRequest-PDU)
int ber_start_constructed(ber_buf_t *b, uint8_t tag, size_t *len_pos);
int ber_end_constructed(ber_buf_t *b, size_t len_pos);

// Decoding
int ber_decode_tlv(const uint8_t *buf, size_t buflen, size_t *off, uint8_t *tag, const uint8_t **val, size_t *vlen);
int ber_decode_integer(const uint8_t *buf, size_t buflen, size_t *off, int64_t *out);
int ber_decode_octet_string(const uint8_t *buf, size_t buflen, size_t *off, const uint8_t **str, size_t *slen);
int ber_decode_oid(const uint8_t *buf, size_t buflen, size_t *off, uint32_t *oid, size_t *oid_len, size_t max_len);

#endif
