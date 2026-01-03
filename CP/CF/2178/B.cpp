// 
#include <bits/stdc++.h>
using namespace std;

int main(){
    int t; 
    cin >> t;
    while(t--){
        string r;
        cin >> r;
        int n = r.size();

        for(char c : r) assert(c == 's' || c == 'u');

        int ans = 0;
        int pos = 0; 

        if(r[pos] == 'u'){
            ans++;
            r[pos] = 's';
        }

        while(pos < n - 1){
            if(pos + 2 < n && r[pos + 2] == 's'){
                pos += 2;
            }
            else if(r[pos + 1] == 's'){
                pos += 1;
            }
            else{
                if(pos + 2 < n){
                    ans++;
                    r[pos + 2] = 's';
                    pos += 2;
                }
                else{
                    ans++;
                    r[pos + 1] = 's';
                    pos += 1;
                }
            }
        }

        cout << ans << '\n';
    }
}
