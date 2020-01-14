

# Add RVM to PATH for scripting. Make sure this is the last PATH variable change.
alias gap='git add --patch'
alias gs='git status'
alias gc='git commit'
alias gcam='git commit --amend'
alias gco='git checkout'
alias gb='git branch'
alias postgres='pg_ctl -D /usr/local/var/postgres'

function get_branch_name {
 git rev-parse --abbrev-ref HEAD
}

alias gp='git push -u origin $(get_branch_name)'
alias gpa='git push -f'


function get_branch_name {
 git rev-parse --abbrev-ref HEAD
}
alias gp='git push -u origin $(get_branch_name)'
alias gpa='git push -f'

function git_branch {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/'
}

parse_git_branch() {
  git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

function git_status {
  git_status=`git status 2> /dev/null`

  if [[ $git_status == *"Changes not staged for commit:"* ]]; then
    echo "*"
  elif [[ $git_status == *"Changes to be committed:"* ]]; then
    echo '+'
  fi
}

function git_color {
  git_status=`git status 2> /dev/null`

  if [[ $git_status == *"nothing to commit, working tree clean"* ]]; then
    echo -e $COLOR_GREEN
  else
    echo -e $COLOR_YELLOW
  fi
}

COLOR_RED="\033[0;31m"
COLOR_YELLOW="\033[0;33m"
COLOR_GREEN="\033[0;32m"
COLOR_OCHRE="\033[38;5;95m"
COLOR_BLUE="\033[36m"
COLOR_ORANGE="\033[0;166m"

COLOR_WHITE="\033[0;37m"
COLOR_RESET="\033[0m"

PS1="\[$COLOR_BLUE\]\u" # username
PS1+="\[$COLOR_BLUE\][\W]" # working directory
PS1+=" \[\$(git_color)\]\$(git_branch)\$(git_status)" # branch info
PS1+="\[$COLOR_RESET\] \$ "
export PS1

export CLICOLOR=1
export LSCOLORS=ExFxBxDxCxegedabagacad


export PATH="$PATH:$HOME/.rvm/bin"

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
