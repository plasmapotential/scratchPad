#include <iostream>
#include <vector>
#include <string>
#include <fstream>


using namespace std;

int main()
{
    char t[] = "0.000010";
    char s[] = "000001";
    cout << t << endl;
//    string fn = "g"+s+"_"+t;
//    cout << fn << endl;

    double time = atof(t);
    int shot = atoi(s);

    string tStr = std::to_string(time);
    string sStr = std::to_string(shot);

    cout << time << endl;
    cout << tStr << endl;

    string fn2 = "g"+sStr+"_"+tStr;
    cout << fn2 << endl;

    return 0;


}


