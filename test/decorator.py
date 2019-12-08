def log(func):
    print(1)
    return func


def now():
    print("2020")

now = log(now)

