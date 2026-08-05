#pragma once

struct noncopyable
{
    noncopyable() = default;
    ~noncopyable() = default;

    // 禁用拷贝
    noncopyable(const noncopyable&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;

    // 默认的移动构造实现
    noncopyable(noncopyable&&) noexcept = default;
    noncopyable& operator=(noncopyable&&) noexcept = default;
};
