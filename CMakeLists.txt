cmake_minimum_required(VERSION 3.30)
project(untitled3 C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Warnings and POSIX feature macros
add_compile_options(-Wall -Wextra -Wpedantic -O2)
add_compile_definitions(_POSIX_C_SOURCE=200112L _DEFAULT_SOURCE)

# Headers are alongside sources
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# Application
add_executable(untitled3
        snmp2otel.c
        ber.c
        oid.c
        snmp.c
        otlp.c
        mapping.c
)

# Tests
include(CTest)
enable_testing()

add_executable(unit_tests
        test_runner.c
        test_oid.c
        test_ber.c
        test_url.c
        ber.c
        oid.c
        otlp.c
)
add_test(NAME unit_tests COMMAND unit_tests)

# Convenience: make run
add_custom_target(run
        COMMAND $<TARGET_FILE:untitled3> -t 127.0.0.1 -o oids.txt -e http://localhost:4318/v1/metrics -v
        DEPENDS untitled3
        USES_TERMINAL
)