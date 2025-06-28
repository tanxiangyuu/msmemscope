// Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

#include "sqlite_loader.h"

int Sqlite3Open(const char* filename, sqlite3** db)
{
    static auto func = Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3OpenFunc>("sqlite3_open");
    if (!func) {
        return SQLITE_ERROR;
    }
    return func(filename, db);
}

int Sqlite3Close(sqlite3* db)
{
    static auto func = Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3CloseFunc>("sqlite3_close");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!db) {
        return SQLITE_MISUSE;
    }
    return func(db);
}

int Sqlite3BusyTimeout(sqlite3* db, int ms)
{
    static auto func =
        Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3BusyTimeoutFunc>("sqlite3_busy_timeout");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!db) {
        return SQLITE_MISUSE;
    }
    return func(db, ms);
}

int Sqlite3Exec(sqlite3* db, const char* sql, int (*callback)(void*, int, char**, char**), void* arg, char** errmsg)
{
    static auto func = Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3ExecFunc>("sqlite3_exec");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!db) {
        return SQLITE_MISUSE;
    }
    return func(db, sql, callback, arg, errmsg);
}

int Sqlite3PrepareV2(sqlite3* db, const char* sql, int nByte, sqlite3_stmt** ppStmt, const char** pzTail)
{
    static auto func =
        Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3PrepareV2Func>("sqlite3_prepare_v2");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!db) {
        return SQLITE_MISUSE;
    }
    return func(db, sql, nByte, ppStmt, pzTail);
}

int Sqlite3Step(sqlite3_stmt* pStmt)
{
    static auto func = Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3StepFunc>("sqlite3_step");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!pStmt) {
        return SQLITE_MISUSE;
    }
    return func(pStmt);
}

int Sqlite3Finalize(sqlite3_stmt* pStmt)
{
    static auto func =
        Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3FinalizeFunc>("sqlite3_finalize");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!pStmt) {
        return SQLITE_MISUSE;
    }
    return func(pStmt);
}

int Sqlite3BindText(sqlite3_stmt* pStmt, int index, const char* value, int n, void(*)(void*))
{
    static auto func =
        Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3BindTextFunc>("sqlite3_bind_text");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!pStmt) {
        return SQLITE_MISUSE;
    }
    if (n == -1 && value == nullptr) {
        return SQLITE_MISUSE;
    }
    return func(pStmt, index, value, n, nullptr);
}

int Sqlite3BindInt(sqlite3_stmt* pStmt, int index, int value)
{
    static auto func =
        Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3BindIntFunc>("sqlite3_bind_int");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!pStmt) {
        return SQLITE_MISUSE;
    }
    return func(pStmt, index, value);
}

int Sqlite3BindInt64(sqlite3_stmt* pStmt, int index, sqlite3_int64 value)
{
    static auto func =
        Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3BindInt64Func>("sqlite3_bind_int64");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!pStmt) {
        return SQLITE_MISUSE;
    }
    return func(pStmt, index, value);
}

const char* Sqlite3Errmsg(sqlite3* db)
{
    static auto func =
        Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3ErrmsgFunc>("sqlite3_errmsg");
    if (!func) {
        return nullptr;
    }
    if (!db) {
        return nullptr;
    }
    return func(db);
}

int Sqlite3Reset(sqlite3_stmt* pStmt)
{
    static auto func = Leaks::VallinaSymbol<Leaks::Sqlite3LibLoader>::Instance().Get<Sqlite3ResetFunc>("sqlite3_reset");
    if (!func) {
        return SQLITE_ERROR;
    }
    if (!pStmt) {
        return SQLITE_MISUSE;
    }
    return func(pStmt);
}