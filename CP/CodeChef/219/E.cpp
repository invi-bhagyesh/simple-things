#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        if (n == 2) { cout << -1 << endl; continue; }
        vector g(n, vector<int>(n, 0));
        g[0][0] = g[0][1] = g[0][2] = g[1][1] = g[1][2] = g[2][1] = g[2][2] = 1;
        for (int j = 3; j < n; j++) g[2][j] = 1;
        for (int i = 3; i < n; i++) g[i][n-1] = 1;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) cout << g[i][j] << " \n"[j == n-1];
    }
}
