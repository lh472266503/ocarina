//
// Created by Zero on 26/04/2022.
//

#include "core/stl.h"

using std::cout;
using std::endl;

struct Flag {};

class var {
public:
    int x;

    var(int x, Flag):x(x) {
        cout << "op" << endl;
    }

    var(int x = 0) : x(x) {
        cout << "assign init" << endl;
    }

    var(var &&other) noexcept {
        x = other.x;
        cout << "move ctor" << endl;
    }

    var(const var &other) {
        cout << "copy" << endl;
        x = other.x;
    }

    var &operator=(const var &other) {
        cout << "assign" << endl;
        x = other.x;
        return *this;
    }

    var &operator=(var &&other) noexcept {
        cout << "assign" << endl;
        x = other.x;
        return *this;
    }

    ~var() {
        cout << "destruct" << endl;
    }
};

var def(int x) {
    return var{x, Flag{}};
}

var operator+(const var &v1, const var &v2) {
    return def(v1.x + v2.x);
}

int main() {

    var a;
    var b;

    var c;
    c = a + b + a;
    cout << "wori" << endl;
}
