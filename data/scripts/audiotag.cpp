#include <iostream>
#include <stdio.h>

#include <string>

#include <taglib/fileref.h>
#include <taglib/tag.h>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3) return 1;

    string what(argv[1]);

    TagLib::FileRef f(argv[2]);

    if(!f.isNull() && f.tag()) {

        TagLib::Tag *tag = f.tag();
       
        if (what == "title")  cout << tag->title()   << endl;
        if (what == "artist") cout << tag->artist()  << endl;
        if (what == "album")  cout << tag->album()   << endl;
        if (what == "year")   cout << tag->year()    << endl;
    }
    return 0;
}

