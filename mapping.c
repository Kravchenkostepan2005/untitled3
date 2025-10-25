#include "mapping.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Minimal, permissive parser for the simple mapping JSON described in the assignment.
// Supports: { "oid": { "name": "...", "unit": "...", "type": "gauge" }, ... }

static int skip_ws(FILE *f)
{
	int c;
	while ((c = fgetc(f)) != EOF) {
		if (c != ' ' && c != '\t' && c != '\r' && c != '\n') return c;
	}
	return EOF;
}

static int read_string(FILE *f, char *buf, size_t cap)
{
	int c = fgetc(f);
	if (c != '"') return -1;
	size_t i = 0;
	while ((c = fgetc(f)) != EOF) {
		if (c == '"') { if (i >= cap) return -1; buf[i] = '\0'; return 0; }
		if (c == '\\') {
			c = fgetc(f); if (c == EOF) return -1;
			if (c == '"' || c == '\\' || c == '/') { /* ok */ }
			else if (c == 'n') c = '\n';
			else if (c == 't') c = '\t';
			else if (c == 'r') c = '\r';
			// no unicode support
		}
		if (i + 1 >= cap) return -1;
		buf[i++] = (char)c;
	}
	return -1;
}

static int expect(FILE *f, char ch)
{
	int c = skip_ws(f);
	if (c != ch) return -1;
	return 0;
}

int mapping_load(const char *path, mapping_table_t *table)
{
	memset(table, 0, sizeof(*table));
	FILE *f = fopen(path, "rb");
	if (!f) return -1;
	if (expect(f, '{') != 0) { fclose(f); return -1; }
	while (1) {
		int c = skip_ws(f);
		if (c == '}') break; // empty
		ungetc(c, f);
		char oid[256]; if (read_string(f, oid, sizeof(oid)) != 0) { fclose(f); return -1; }
		if (expect(f, ':') != 0) { fclose(f); return -1; }
		if (expect(f, '{') != 0) { fclose(f); return -1; }
		char name[128] = ""; char unit[32] = ""; char key[64]; char val[256];
		while (1) {
			int c2 = skip_ws(f);
			if (c2 == '}') break;
			ungetc(c2, f);
			if (read_string(f, key, sizeof(key)) != 0) { fclose(f); return -1; }
			if (expect(f, ':') != 0) { fclose(f); return -1; }
			if (strcmp(key, "name") == 0 || strcmp(key, "unit") == 0 || strcmp(key, "type") == 0) {
				if (read_string(f, val, sizeof(val)) != 0) { fclose(f); return -1; }
				if (strcmp(key, "name") == 0) strncpy(name, val, sizeof(name)-1);
				else if (strcmp(key, "unit") == 0) strncpy(unit, val, sizeof(unit)-1);
				// type must be gauge if present; ignore validation to be permissive
			} else {
				// skip value (string or number)
				int ch = skip_ws(f);
				if (ch == '"') { ungetc(ch, f); if (read_string(f, val, sizeof(val)) != 0) { fclose(f); return -1; } }
				else { // read until comma or }
					while (ch != ',' && ch != '}' && ch != EOF) ch = fgetc(f);
					if (ch == '}') { ungetc(ch, f); }
				}
			}
			// comma between fields or end
			int sep = skip_ws(f);
			if (sep == ',') continue; else if (sep == '}') break; else { fclose(f); return -1; }
		}
		// add entry
		mapping_entry_t *ents = realloc(table->entries, (table->count + 1) * sizeof(mapping_entry_t));
		if (!ents) { fclose(f); return -1; }
		table->entries = ents;
		strncpy(table->entries[table->count].oid, oid, sizeof(table->entries[table->count].oid)-1);
		strncpy(table->entries[table->count].name, name, sizeof(table->entries[table->count].name)-1);
		strncpy(table->entries[table->count].unit, unit, sizeof(table->entries[table->count].unit)-1);
		table->count++;
		// comma between items or end
		int sep2 = skip_ws(f);
		if (sep2 == ',') continue; else if (sep2 == '}') break; else { fclose(f); return -1; }
	}
	fclose(f);
	return 0;
}

void mapping_free(mapping_table_t *table)
{
	free(table->entries);
	table->entries = NULL; table->count = 0;
}

const mapping_entry_t *mapping_find(const mapping_table_t *table, const char *oid)
{
	for (size_t i = 0; i < table->count; ++i) {
		if (strcmp(table->entries[i].oid, oid) == 0) return &table->entries[i];
	}
	return NULL;
}
