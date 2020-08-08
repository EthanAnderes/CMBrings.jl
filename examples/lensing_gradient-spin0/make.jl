#=

```shell
julia make.jl
jupyter nbconvert --to notebook --execute --inplace note.ipynb
```

The first line converts src.jl to a notebook.
The second line runs the notebook.

If you want to save that in another format do the following 

jupyter nbconvert --to html     note.ipynb --output note.html
jupyter nbconvert --to markdown note.ipynb
jupyter nbconvert --to webpdf   note.ipynb
```

=#

using Literate                 

config = Dict(                 
    "name"          => "note",
    "documenter"    => false,  
    "keep_comments" => false,  
    "execute"       => false, 
    "credit"        => false,  
) 

Literate.notebook("src.jl"; config=config)                              
