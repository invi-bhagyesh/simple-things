#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;

    vector<pair<int,int>> st;

    for (int i = 0; i < n; i++) {
        int x;
        cin >> x;

        if (!st.empty() && st.back().first == x) {
            st.back().second++;
            if (st.back().second == 4) {
                st.pop_back();
            }
        } else {
            st.push_back({x, 1});
        }
    }

    int ans = 0;
    for (auto &p : st) ans += p.second;

    cout << ans << endl;
}
