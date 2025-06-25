// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#ifndef SQLITE_LOADER_H
#define SQLITE_LOADER_H

#include <cstdint>
#include "vallina_symbol.h"

using sqlite3 = struct sqlite3;
using sqlite3_stmt = struct sqlite3_stmt;
using sqlite3_int64 = std::int64_t;
using sqlite3_destructor_type = void(*)(void*);
constexpr sqlite3_destructor_type SQLITE_STATIC = reinterpret_cast<sqlite3_destructor_type>(0);
constexpr int SQLITE_OK = 0;
constexpr int SQLITE_ERROR = 1;
constexpr int SQLITE_MISUSE = 21;
constexpr int SQLITE_ROW = 100;
constexpr int SQLITE_DONE = 101;


using Sqlite3OpenFunc = int (*)(const char*, sqlite3**);
using Sqlite3CloseFunc = int (*)(sqlite3*);
using Sqlite3ExecFunc = int (*)(sqlite3*, const char*, int(*)(void*, int, char**, char**), void*, char**);
using Sqlite3PrepareV2Func = int (*)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
using Sqlite3StepFunc = int (*)(sqlite3_stmt*);
using Sqlite3FinalizeFunc = int (*)(sqlite3_stmt*);
using Sqlite3BindTextFunc = int (*)(sqlite3_stmt*, int, const char*, int, void(*)(void*));
using Sqlite3BindIntFunc = int (*)(sqlite3_stmt*, int, int);
using Sqlite3BindInt64Func = int (*)(sqlite3_stmt*, int, sqlite3_int64);
using Sqlite3BusyTimeoutFunc = int (*)(sqlite3*, int);
using Sqlite3ErrmsgFunc = const char* (*)(sqlite3*);
using Sqlite3ResetFunc = int (*)(sqlite3_stmt*);

int Sqlite3Open(const char* filename, sqlite3** db);
int Sqlite3Close(sqlite3* db);
int Sqlite3BusyTimeout(sqlite3* db, int ms);
int Sqlite3Exec(sqlite3* db, const char* sql, int (*callback)(void*, int, char**, char**), void* arg, char** errmsg);
int Sqlite3PrepareV2(sqlite3* db, const char* sql, int nByte, sqlite3_stmt** ppStmt, const char** pzTail);
int Sqlite3Step(sqlite3_stmt* pStmt);
int Sqlite3Finalize(sqlite3_stmt* pStmt);
int Sqlite3BindText(sqlite3_stmt* pStmt, int index, const char* value, int n, void(*)(void*));
int Sqlite3BindInt(sqlite3_stmt* pStmt, int index, int value);
int Sqlite3BindInt64(sqlite3_stmt* pStmt, int index, sqlite3_int64 value);
const char* Sqlite3Errmsg(sqlite3* db);
int Sqlite3Reset(sqlite3_stmt* pStmt);
#endif