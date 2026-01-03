#include <bits/stdc++.h>
using namespace std;


int main() {
    int t; int INF = 1e9;
    cin >> t;
    while (t--) {
        int n;
        string s;
        cin >> n >> s;
        int A = INF;
        for (int i = 0; i + 3 < n; i++)
            A = min(A,
                (s[i] != '2') + (s[i+1] != '0') + (s[i+2] != '2') + (s[i+3] != '6')
            );

        vector<vector<int>> dp(n+1, vector<int>(4, INF));
        dp[0][0] = 0;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) {
                if (dp[i][j] == INF) continue;
                for (char c : {'0','2','5','6'}) {
                    int k = 0;

                    if (j == 0) {
                        if (c == '2') k = 1;
                    }
                    else if (j == 1) {
                        if (c == '0') k = 2;
                        else if (c == '2') k = 1;
                    }
                    else if (j == 2) {
                        if (c == '2') k = 3;
                        else if (c == '0') k = 0;
                        else k = 0;
                    }
                    else if (j == 3) {
                        if (c == '5') continue; 
                        else if (c == '2') k = 1;
                        else if (c == '0') k = 2;  
                        else k = 0;
                    }

                    dp[i+1][k] = min(dp[i+1][k],
                        dp[i][j] + (s[i] != c));
                }
            }
        }

        int B = *min_element(dp[n].begin(), dp[n].end());
        cout << min(A, B) <<endl;
    }
}
