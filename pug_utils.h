#include <string>

namespace pug_utils {

    bool is_number(const std::string& s)
    {
        return !s.empty() && std::find_if(s.begin(), 
        s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
    }

};
