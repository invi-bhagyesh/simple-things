// stack or union?? union i guessa
#include <bits/stdc++.h>
using namespace std;
typedef long long ll; 

int f[100005], s[100005];
int g(int x) { return f[x] == x ? x : f[x] = g(f[x]); }

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        vector<int> p(n), d(n - 1);
        for (int i = 0; i < n; i++) cin >> p[i];
        for (int i = 0; i < n - 1; i++) d[i] = abs(p[i] - p[i + 1]);
        
        vector<vector<int>> w(n + 1);
        for (int i = 0; i < n - 1; i++) w[d[i]].push_back(i);
        
        for (int i = 0; i < n - 1; i++) f[i] = i, s[i] = 1;
        vector<bool> v(n - 1, false);
        
        auto c = [](int x) { return (ll)x * (x + 1) / 2; };
        
        vector<ll> r(n);
        ll u = 0;
        for (int k = n - 1; k >= 1; k--) {
            for (int i : w[k]) {
                v[i] = true;
                u += 1;
                if (i > 0 && v[i - 1]) {
                    int a = g(i), b = g(i - 1);
                    u -= c(s[a]) + c(s[b]);
                    f[a] = b;
                    s[b] += s[a];
                    u += c(s[b]);
                }
                if (i < n - 2 && v[i + 1]) {
                    int a = g(i), b = g(i + 1);
                    u -= c(s[a]) + c(s[b]);
                    f[a] = b;
                    s[b] += s[a];
                    u += c(s[b]);
                }
            }
            r[k] = u;
        }
        
        for (int k = 1; k < n; k++)
            cout << r[k] << " \n"[k == n - 1];
    }
}
