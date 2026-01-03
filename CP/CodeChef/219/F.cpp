#include <bits/stdc++.h>
using namespace std;
using ll = long long;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;vector<int> pos;
        string s;
        cin >> n >> s;
        int o = count(s.begin(), s.end(), '1'), z = n - o;
        int max_F = (o >= z) ? n : 2 * o;
        for (int i = 0; i < n; i++) if (s[i] == '1') pos.push_back(i);
        
        ll sw = 0;
        int p = -1;
        for (int j = 0; j < o; j++) {
            int tar = max(min(pos[j], 2 * j), p + 1);
            sw += pos[j] - tar;
            p = tar;
        }
        cout << max_F << " " << sw << endl;
    }
}
