/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#include "ber.h"
#include <string.h>

static size_t encode_length(uint8_t *out, size_t len)
{
	if (len < 128) {
		out[0] = (uint8_t)len;
		return 1;
	}
	uint8_t tmp[8];
	int i = 0;
	while (len > 0) {
		tmp[i++] = (uint8_t)(len & 0xFF);
		len >>= 8;
	}
	out[0] = 0x80 | (uint8_t)i;
	for (int j = 0; j < i; ++j) out[1 + j] = tmp[i - 1 - j];
	return 1 + (size_t)i;
}

static int decode_length(const uint8_t *buf, size_t buflen, size_t *off, size_t *len)
{
	if (*off >= buflen) return -1;
	uint8_t b = buf[(*off)++];
	if ((b & 0x80) == 0) {
		*len = b;
		return 0;
	}
	int n = b & 0x7F;
	if (n == 0 || n > 8) return -1;
	if (*off + (size_t)n > buflen) return -1;
	size_t v = 0;
	for (int i = 0; i < n; ++i) v = (v << 8) | buf[(*off)++];
	*len = v;
	return 0;
}

void ber_buf_init(ber_buf_t *b, uint8_t *storage, size_t cap)
{
	b->data = storage;
	b->len = 0;
	b->cap = cap;
}

size_t ber_buf_write(ber_buf_t *b, const void *src, size_t n)
{
	if (b->len + n > b->cap) return 0;
	memcpy(b->data + b->len, src, n);
	b->len += n;
	return n;
}

int ber_encode_tlv(ber_buf_t *b, uint8_t tag, const uint8_t *value, size_t vlen)
{
	uint8_t lenbuf[9];
	size_t l = encode_length(lenbuf, vlen);
	if (b->len + 1 + l + vlen > b->cap) return -1;
	b->data[b->len++] = tag;
	memcpy(b->data + b->len, lenbuf, l);
	b->len += l;
	memcpy(b->data + b->len, value, vlen);
	b->len += vlen;
	return 0;
}

int ber_encode_integer(ber_buf_t *b, int64_t value)
{
	uint8_t tmp[9];
	int neg = value < 0;
	uint64_t v = (uint64_t)(value);
	int i = 0;
	do {
		tmp[i++] = (uint8_t)(v & 0xFF);
		v >>= 8;
	} while ((neg && (v != 0xFFFFFFFFFFFFFFFFULL || (tmp[i-1] & 0x80) == 0)) || (!neg && (v != 0 || (tmp[i-1] & 0x80))));
	for (int j = 0; j < i / 2; ++j) {
		uint8_t t = tmp[j]; tmp[j] = tmp[i-1-j]; tmp[i-1-j] = t;
	}
	return ber_encode_tlv(b, 0x02, tmp, (size_t)i);
}

int ber_encode_octet_string(ber_buf_t *b, const uint8_t *str, size_t slen)
{
	return ber_encode_tlv(b, 0x04, str, slen);
}

int ber_encode_oid(ber_buf_t *b, const uint32_t *oid, size_t oid_len)
{
	if (oid_len < 2) return -1;
	uint8_t tmp[256];
	size_t pos = 0;
	tmp[pos++] = (uint8_t)(oid[0] * 40 + oid[1]);
	for (size_t i = 2; i < oid_len; ++i) {
		uint32_t x = oid[i];
		uint8_t stack[8];
		int sp = 0;
		do { stack[sp++] = (uint8_t)(x & 0x7F); x >>= 7; } while (x);
		for (int j = sp - 1; j >= 0; --j) {
			uint8_t b7 = stack[j];
			if (j != 0) b7 |= 0x80;
			tmp[pos++] = b7;
			if (pos >= sizeof(tmp)) return -1;
		}
	}
	return ber_encode_tlv(b, 0x06, tmp, pos);
}

// Generic constructed (arbitrary tag)
int ber_start_constructed(ber_buf_t *b, uint8_t tag, size_t *len_pos)
{
	if (b->len + 2 > b->cap) return -1;
	b->data[b->len++] = tag;
	*len_pos = b->len;
	b->data[b->len++] = 0x80; // placeholder for length
	return 0;
}

int ber_end_constructed(ber_buf_t *b, size_t len_pos)
{
	size_t start = len_pos + 1;
	size_t content_len = b->len - start;
	uint8_t lenbuf[9];
	size_t l = encode_length(lenbuf, content_len);
	if (l == 1) {
		b->data[len_pos] = lenbuf[0];
		return 0;
	}
	if (b->len + (l - 1) > b->cap) return -1;
	memmove(b->data + start + (l - 1), b->data + start, content_len);
	b->data[len_pos] = lenbuf[0];
	memcpy(b->data + len_pos + 1, lenbuf + 1, l - 1);
	b->len += (l - 1);
	return 0;
}

// SEQUENCE wrappers via generic constructed
int ber_start_sequence(ber_buf_t *b, size_t *len_pos)
{
	return ber_start_constructed(b, 0x30, len_pos);
}

int ber_end_sequence(ber_buf_t *b, size_t len_pos)
{
	return ber_end_constructed(b, len_pos);
}

int ber_decode_tlv(const uint8_t *buf, size_t buflen, size_t *off, uint8_t *tag, const uint8_t **val, size_t *vlen)
{
	if (*off + 2 > buflen) return -1;
	*tag = buf[(*off)++];
	size_t len;
	if (decode_length(buf, buflen, off, &len) != 0) return -1;
	if (*off + len > buflen) return -1;
	*val = buf + *off;
	*vlen = len;
	*off += len;
	return 0;
}

int ber_decode_integer(const uint8_t *buf, size_t buflen, size_t *off, int64_t *out)
{
	uint8_t tag; const uint8_t *v; size_t vlen;
	if (ber_decode_tlv(buf, buflen, off, &tag, &v, &vlen) != 0) return -1;
	if (tag != 0x02 || vlen == 0 || vlen > 8) return -1;
	int64_t val = (v[0] & 0x80) ? -1 : 0;
	for (size_t i = 0; i < vlen; ++i) val = (val << 8) | v[i];
	*out = val;
	return 0;
}

int ber_decode_octet_string(const uint8_t *buf, size_t buflen, size_t *off, const uint8_t **str, size_t *slen)
{
	uint8_t tag; const uint8_t *v; size_t vlen;
	if (ber_decode_tlv(buf, buflen, off, &tag, &v, &vlen) != 0) return -1;
	if (tag != 0x04) return -1;
	*str = v; *slen = vlen; return 0;
}

int ber_decode_oid(const uint8_t *buf, size_t buflen, size_t *off, uint32_t *oid, size_t *oid_len, size_t max_len)
{
	uint8_t tag; const uint8_t *v; size_t vlen;
	if (ber_decode_tlv(buf, buflen, off, &tag, &v, &vlen) != 0) return -1;
	if (tag != 0x06 || vlen == 0) return -1;
	if (max_len < 2) return -1;
	size_t pos = 0; size_t i = 0;
	uint8_t first = v[pos++];
	oid[i++] = first / 40; oid[i++] = first % 40;
	while (pos < vlen && i < max_len) {
		uint32_t val = 0;
		while (pos < vlen) {
			uint8_t b = v[pos++];
			val = (val << 7) | (b & 0x7F);
			if ((b & 0x80) == 0) break;
		}
		oid[i++] = val;
	}
	*oid_len = i;
	return 0;
}
