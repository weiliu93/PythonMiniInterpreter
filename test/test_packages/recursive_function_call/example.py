def fibo(n):
    if n <= 1:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)


print(fibo(10))


dp = {}


def fibo_with_dp(n):
    if n <= 1:
        return 1
    else:
        if n in dp:
            return dp[n]
        else:
            dp[n] = fibo_with_dp(n - 1) + fibo_with_dp(n - 2)
            return dp[n]


print(fibo_with_dp(100))
