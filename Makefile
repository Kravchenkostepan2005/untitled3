# Simple, portable Makefile for snmp2otel (sources in project root)

CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2 -g -D_POSIX_C_SOURCE=200809L -D_DEFAULT_SOURCE
LDFLAGS ?=

# Application sources (main + libs)
APP_SRCS := ber.c oid.c snmp.c otlp.c mapping.c snmp2otel.c
APP_OBJS := $(APP_SRCS:.c=.o)
BIN := snmp2otel

.PHONY: all run test clean

# Build the application
all: $(BIN)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN): $(APP_OBJS)
	$(CC) $(CFLAGS) $(APP_OBJS) -o $@ $(LDFLAGS)

# Run the app with custom arguments: make run ARGS='...'
run: $(BIN)
	./$(BIN) $(ARGS) || true

# ----------------------------
# Tests (test_*.c in project root)
# ----------------------------
# Test sources (runner + unit tests)
TEST_SRCS := test_runner.c test_oid.c test_ber.c test_url.c
# Library sources (everything except the main)
LIB_SRCS  := $(filter-out snmp2otel.c,$(APP_SRCS))
TEST_BIN  := run_tests

# Build and run tests
test: CFLAGS += -g -O0
test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(LIB_SRCS) $(TEST_SRCS)
	$(CC) $(CFLAGS) $(LIB_SRCS) $(TEST_SRCS) -o $@ $(LDFLAGS)

# Clean build artifacts
clean:
	rm -f $(APP_OBJS) $(BIN) $(TEST_BIN)