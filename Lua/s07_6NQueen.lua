#!/usr/bin/env lua

lanes = require "lanes".configure()

local function loop1( )
  while true do
    print("loop1")
    os.execute("sleep 0.1")
  end
end

function loop2()
  while true do
    print("loop2")
    os.execute("sleep 0.1")
  end
end


thread1= lanes.gen("*",loop1)()
thread2= lanes.gen("*",loop2)()

-- lua lanesのインストール
-- luajitでは動かない。luaに戻す macデフォルトのluaをアンインストール
-- sudo port uninstall lua

-- luaのインストールまたはアップグレード
-- $ brew [install/upgrade] lua
-- $ brew install lurrocks

-- lanesのインストール(両方入れる必要有り）
-- $ luarocks install lanes
-- $ luarocks --local install lanes

-- lanesライブラリの格納場所を作成
-- $ mkdir ~/.luarocks

-- ~/.bash_profileにPATHを通す
-- $ vim ~/.bash_profile
-- export LUA_PATH='~/.luarocks/share/lua/5.2/lanes.lua;;'
-- export LUA_CPATH='~/.luarocks/lib/lua/5.2/lanes/core.so;;'

-- $./s07_6NQueen.lua
--
