file(GLOB_RECURSE SOURCE "*.cpp"
${EVENT_TRACE_INC_DIR}/event_report.cpp
${ANALYSIS_DIR}/dump.cpp
)
list(SORT SOURCE)

target_sources(ascend_leaks PRIVATE ${SOURCE})
target_include_directories(ascend_leaks PUBLIC
  ${UTILITY_INC_DIR}
  ${ANALYSIS_DIR}
  ${JSON_DIR}
)
