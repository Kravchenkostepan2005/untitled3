#ifndef MAPPING_H
#define MAPPING_H

#include <stddef.h>

typedef struct {
    char oid[256];
    char name[128];
    char unit[32];
} mapping_entry_t;

typedef struct {
    mapping_entry_t *entries;
    size_t count;
} mapping_table_t;

// Load optional mapping JSON file. Returns 0 on success, -1 on parse or IO error.
int mapping_load(const char *path, mapping_table_t *table);
void mapping_free(mapping_table_t *table);
// Lookup by OID string; returns pointer or NULL
const mapping_entry_t *mapping_find(const mapping_table_t *table, const char *oid);

#endif
