#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int d, f;
    cin>>d>>f;
    

    int t = (f -  (d % 7) + 7) % 7;
    if (t == 0) t = 7;
    cout << t <<endl;
}

