# Small Graph Lib

## Installing

    $ git clone https://github.com/wxgeo/smallgraphlib

    $ pip install --user smallgraphlib

## Usage

Main class is `Graph`:

    >>> from smallgraphlib import Graph
    >>> g = Graph(["A", "B", "C"], ("A", "B"), ("B", "A"), ("B", "C"))
    >>> g.is_simple
    True
    >>> g.is_complete
    False
    >>> g.is_directed
    True
    >>> g.adjacency_matrix
    [[0, 1, 0], [1, 0, 1], [0, 0, 0]]
    >>> g.degree
    3
    >>> g.order
    3

    
For the graph is not to complesikz code may be generated:

## Writing a new script

### Guidelines

Each script should declare a new test class inheriting from class `WebsiteTest`.

This subclass should implement a `.get_authors(self) -> List[str]` method, 
which must return the list of the authors for each website.

It may also add tests. Tests are `Website` methods decorated with `@test(title: str, coeff: float)`.

Each test must return two values:
- the score, which must be a float between 0 and 1
- the log, which must be a list of strings (possibly empty)

The main class for running tests is `CollectTestsResults`.
It takes two arguments:
- `paths` is a list of paths (strings or `Path` instances)
- `test_class` is the class to be used for the test, i.e. the custom class
  you wrote inheriting from WebsiteTest.

It has 3 main methods:
- `.run()` will return the tests results as a dict.
- `.write_log(folder_path)` will create a `path/log/` folder, with a `.md` file 
  for each author.
- `.write_xlsx_file(file_path)` will create an XLSX (Excel) file
   will all results (`file_path` may be omitted).

However, you may simply call `run_and_collect()` function for convenience.


### Full example

1. Create file `mytest.py`:


        from websites_test_framework import Website, CollectTestsResults, test
    
        PATH = "/websites/parent/directory"
        # PATH may also be a glob, like "/home/*/*/www" or "/home/**/www".
    
        class MyWebsiteTest(WebsiteTest):
            def get_authors(self):
                with open(self.path / "authors.txt") as file:
                    return [line.strip() for line in file]
    
            @test("css folder ?", coeff=0.5)
            def test_css_folder(self):
                if (self.path / "css").is_dir():
                    return 1, []
                elif (self.path / "styles").is_dir():
                    return 0.5, ["folder `styles` should be named `css`"]:
                else:
                    return 0, ["folder css/ not found !"]
    
        if __file__ == "__main__":
            # Write logs and output XLSX file in current directory.
            run_and_collect(PATH, MyWebsiteTest)

2. Launch tests:


        $ python3 mytest.py


3. Results are exported to `scores.xlsx` by default.
 

        $ libreoffice scores.xlsx
