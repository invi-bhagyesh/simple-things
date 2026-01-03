#include <bits/stdc++.h>
using namespace std;

int main(){
    int n;
    cin>>n;
    vector<int> x(n), y(n);
    for (int &t:x){cin>>t;}
    for (int &t:y){cin>>t;}

    int max_s = 0;

    for (int i=0;i<n;i++){
        for (int j=0; j<n;j++){
            int dx = x[i]-x[j];
            int dy = y[i]-y[j];

            int diff = dx*dx + dy*dy;
            max_s = max(max_s, diff);
        }
    }

    cout<< max_s<<endl;
}