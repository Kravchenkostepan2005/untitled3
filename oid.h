/*
 * Author: Stepan Kravchenko (xkravc03)
 */

#ifndef OID_H
#define OID_H

#include <stddef.h>
#include <stdint.h>

// Parse numeric dotted OID (e.g., "1.3.6.1.2.1.1.3.0") into array of uint32_t
// Returns number of components on success (>0), or 0 on error.
size_t oid_parse(const char *s, uint32_t *out, size_t max_len);

// Convert array to dotted string into provided buffer. Returns number of chars written (excluding NUL), or 0 on error.
size_t oid_to_string(const uint32_t *oid, size_t len, char *buf, size_t bufsize);

#endif
