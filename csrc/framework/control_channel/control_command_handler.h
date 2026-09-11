/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#ifndef CONTROL_COMMAND_HANDLER_H
#define CONTROL_COMMAND_HANDLER_H

#include <memory>
#include <string>

#include "analysis/analyzer_base.h"
#include "analysis/event.h"

namespace MemScope
{

/*
 * 控制字命令处理器(业务进程侧),职责分两层:
 *  HandleDispatch: 派发内(dispatcher锁内)仅白名单校验+标记,不执行命令逻辑
 *     (派发内执行会重入dispatcher非递归锁);
 *  Execute: 派发返回后(锁已释放)由监听线程执行命令并写ev->output。
 * 两侧独立执行同一白名单校验(Execute侧为防御性兜底)。命令集:start/stop/step;
 * display hook/analyzer/config/memory summary/block、host_leak summary;set config。
 * 继承AnalyzerBase仅为复用派发器订阅框架与GetName约束(display analyzer按名字
 * 过滤自身),不产生控制面能力声明。
 */
class ControlCommandHandler : public AnalyzerBase
{
   public:
    static ControlCommandHandler& GetInstance();

    // 派发内:仅登记(白名单校验+标记);不执行任何业务
    void HandleDispatch(std::shared_ptr<EventBase>& event, MemoryState* state);
    // 派发外:执行命令写ev->output(监听线程串行调用,命令内部各自加锁)
    void Execute(std::shared_ptr<ControlEvent>& event);

    // AnalyzerBase约束(不实际进入事件派发主线,display analyzer过滤自身用)
    void EventHandle(std::shared_ptr<EventBase>& event, MemoryState* state) override;
    const char* GetName() const override { return "control_channel"; }

   private:
    ControlCommandHandler() = default;
    ~ControlCommandHandler() override = default;
    ControlCommandHandler(const ControlCommandHandler&) = delete;
    ControlCommandHandler& operator=(const ControlCommandHandler&) = delete;

    // 各控制字实现(全部写ev->output,不抛异常)
    void DoStart(std::shared_ptr<ControlEvent>& ev);
    void DoStop(std::shared_ptr<ControlEvent>& ev);
    void DoStep(std::shared_ptr<ControlEvent>& ev);
    void DoDisplayHook(std::shared_ptr<ControlEvent>& ev);
    void DoDisplayAnalyzer(std::shared_ptr<ControlEvent>& ev);
    void DoDisplayConfig(std::shared_ptr<ControlEvent>& ev);
    void DoDisplayMemorySummary(std::shared_ptr<ControlEvent>& ev);
    void DoDisplayMemoryBlock(std::shared_ptr<ControlEvent>& ev);
    void DoDisplayHostLeakSummary(std::shared_ptr<ControlEvent>& ev);
    void DoSetConfig(std::shared_ptr<ControlEvent>& ev);
};

}  // namespace MemScope

#endif
