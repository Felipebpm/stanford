" === VUNDLE STUFF

set nocompatible              " be iMproved, required
filetype off                  " required
set tabstop=2
set shiftwidth=2
set expandtab
set number
set hlsearch
set incsearch
set numberwidth=6
" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" ==== plugin manager
Plugin 'VundleVim/Vundle.vim'

" Code completion
Plugin 'Valloric/YouCompleteMe'
Plugin 'SirVer/ultisnips'
Plugin 'honza/vim-snippets'

" ==== Search
"Plugin 'ggreer/the_silver_searcher'
Plugin 'mileszs/ack.vim'
Plugin 'junegunn/fzf.vim'

" ==== Formatting
Plugin 'google/vim-maktaba'
Plugin 'google/vim-codefmt'

" ==== Import Optimization
Plugin 'tell-k/vim-autoflake'

" === Window swap
Plugin 'https://github.com/wesQ3/vim-windowswap'

" === linter 
Plugin 'https://github.com/ngmy/vim-rubocop'
Plugin 'w0rp/ale'
Plugin 'vim-airline/vim-airline' 

" ==== Folding Code
Plugin 'tmhedberg/SimpylFold'

" === Latex Plugin
Plugin 'lervag/vimtex'

" ==== File tree
Plugin 'scrooloose/nerdtree'

" ==== Git
Plugin 'airblade/vim-gitgutter'
Plugin 'tpope/vim-fugitive'

" ==== syntax helpers
Plugin 'bhurlow/vim-parinfer'
Plugin 'scrooloose/syntastic'
Plugin 'scrooloose/nerdcommenter'
Plugin 'tpope/vim-surround'
Plugin 'cakebaker/scss-syntax.vim'
Plugin 'othree/yajs.vim'
Plugin 'mitsuhiko/vim-jinja'
Plugin 'octol/vim-cpp-enhanced-highlight'
Plugin 'ap/vim-css-color'
Plugin 'Vimjas/vim-python-pep8-indent'
Plugin 'python-rope/rope'
Plugin 'python-rope/ropemode'
Plugin 'python-rope/ropevim'
Plugin 'klen/pylama'

" ==== moving / searching
Plugin 'easymotion/vim-easymotion'
Plugin 'kien/ctrlp.vim'
Plugin 'ervandew/supertab'
Plugin 'terryma/vim-multiple-cursors'

" ==== colors
Plugin 'sheerun/vim-polyglot'
Plugin 'joshdick/onedark.vim'

" ====================
" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ

" ==== NERDTREE
let NERDTreeIgnore = ['__pycache__', '\.pyc$', '\.o$', '\.so$', '\.a$', '\.swp', '*\.swp', '\.swo', '\.swn', '\.swh', '\.swm', '\.swl', '\.swk', '\.sw*$', '[a-zA-Z]*egg[a-zA-Z]*', '.DS_Store']

" ==== SimplyFold
nnoremap <space> za
let g:SimpylFold_docstring_preview = 1

"autocmd vimenter * NERDTree
let NERDTreeShowHidden=1
let g:NERDTreeWinPos="left"
let g:NERDTreeDirArrows=0
map <C-t> :NERDTreeToggle<CR>

syntax on
" Color Themes

"---ARCADIA

"DARKER GRAY
"let g:arcadia_Sunset = 1
"colorscheme arcadia

"DARKEST GRAY
"let g:arcadia_Twilight = 1
"colorscheme arcadia

"ALMOST BLACK
"let g:arcadia_Midnight = 1
"colorscheme arcadia

"BLACK
"let g:arcadia_Pitch = 1
"colorscheme arcadia

"---SIERRA

"DARKER GRAY
"let g:sierra_Sunset = 1
"colorscheme sierra
"
"DARKEST GRAY
"let g:sierra_Twilight = 1
"colorscheme sierra
"
"ALMOST BLACK
"let g:sierra_Midnight = 1
"colorscheme sierra
"
"BLACK
"let g:sierra_Pitch = 1
"colorscheme sierra
"

colorscheme despacio

" === Arrow key as ->
imap <S-Right> ->

let g:ctrlp_map = '<c-p>'
let g:ctrlp_cmd = 'CtrlP'

" === Commenter
" Add spaces after comment delimiters by default
let g:NERDSpaceDelims = 1

" Use compact syntax for prettified multi-line comments
let g:NERDCompactSexyComs = 1

" Add your own custom formats or override the defaults
let g:NERDCustomDelimiters = { 'c': { 'left': '/**','right': '*/' } }

" Allow commenting and inverting empty lines (useful when commenting a region)
let g:NERDCommentEmptyLines = 1

" Enable NERDCommenterToggle to check all selected lines is commented or not 
let g:NERDToggleCheckAllLines = 1

" === Search
let g:ackprg = 'ag --nogroup --nocolor --column'
let g:ctrlp_user_command = ['.git/', 'git --git-dir=%s/.git ls-files -oc --exclude-standard']

map <C-l> <leader>ci
let g:vimrubocop_keymap = 0
nmap <Leader>r :RuboCop<CR>
map <C-k> <Up>ddp<Up>
map <C-j> ddp
" Set specific linters
let g:ale_linters = {
\   'javascript': ['eslint'],
\   'ruby': ['rubocop'],
\}
let g:ale_linters_explicit = 1
" let g:ale_set_highlights = 0 
let g:ale_sign_column_always = 1

" Airline
let g:airline#extensions#ale#enabled = 1
" let g:airline#extensions#tabline#enabled = 1
let g:UltiSnipsExpandTrigger="<C-<space>>"
let g:UltiSnipsJumpForwardTrigger="<C-b>"
let g:UltiSnipsJumpBackwardTrigger="<C-z>"

autocmd FileType javascript set formatprg=prettier\ --stdin

" === NUMBER HIGHLIGHTING
highlight LineNr term=bold cterm=NONE ctermfg=DarkGrey ctermbg=NONE gui=NONE guifg=DarkGrey guibg=NONE
