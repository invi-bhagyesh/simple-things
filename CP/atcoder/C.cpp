#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N;
    cin >> N;

    int prev = -1, run = 0;
    ll ans = 0;

    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        if (x == prev) {
            run++;
        } else {
            ans += run % 4;
            run = 1;
            prev = x;
        }
    }

    ans += run % 4;
    cout << ans << endl;
}
