#!/usr/bin/env ruby

################################################################################
# default
################################################################################
@n = 3

################################################################################
# Arguments
################################################################################
require "optparse"
OptionParser.new { |opts|
  # options
  opts.on("-h","--help","Show this message") {
    puts opts
    exit
  }
  opts.on("-n [int]"){ |f|
    @n = f.to_i
  }
  # parse
  opts.parse!(ARGV)
}

################################################################################
# classes
################################################################################
########################################
# grid graph
########################################
class GridGraph
  #### new ####
  def initialize(n)
    @n = n
    @e = []
    id = 0
    e = [0, 1]

    for i in 1..2 * (@n-1)
      if i < @n
        m = 2 * i
      else
        m = 2 * (2*(@n-1) - i + 1)
      end

      for j in 1..m
        if j == 1
          e[0] += 1
          e[1] += 1
        elsif (i < @n && j % 2 == 0) || (i >= @n && j % 2 == 1)
          e[1] += 1
        else
          e[0] += 1
        end

        #
        @e.push(e.clone)
        id += 1
      end
    end

    #### return edges ####
    def edge
      @e
    end
  end
end

########################################
# mate
########################################
class Mate
  #### new ####
  def initialize
    @mate = Hash.new(nil)
  end

  #### size ####
  def size
    @mate.size
  end

  #### look a value of i ####
  def look(i)
    @mate[i]
  end

  #### about node ####
  def add_node(i)
    @mate[i] = i if @mate[i] == nil
  end
  def delete_node(i)
    v =  @mate[i]
    @mate.delete(i)
    if v == nil || v == 0 || v == i
      true
    else
      false
    end
  end

  #### about edge ####
  def add_edge(e, v)
    #### add as 0 ####
    return true if v == 0

    #### add as 1 ####
    i = e[0]
    j = e[1]

    #### unaddable ####
    return false if @mate[i] == 0 || @mate[j] == 0 # digree > 2
    return false if @mate[i] == j && @mate[j] == i # cycle

    #### addable ####
    add_node(i)
    add_node(j)

    # new sub-path
    if @mate[i] == i && @mate[j] == j
      @mate[i] = j
      @mate[j] = i

    # extend sub-path
    elsif @mate[i] == i || @mate[j] == j
      if @mate[j] == j
        t = i;  i = j;  j = t
      end
      s = @mate[j]
      @mate[i] = s
      @mate[s] = i
      @mate[j] = 0

    # conect two sub-paths
    else
      s = @mate[i]
      t = @mate[j]
      @mate[s] = t
      @mate[t] = s
      @mate[i] = 0
      @mate[j] = 0
    end

    return true
  end

  #### to_key ####
  def to_key
    @mate.keys.sort.map{|key| "#{key}:#{@mate[key]};" }.join
  end

  #### set hash [key, val] ####
  def set(key, val)
    @mate[key] = val
  end

  #### clone ####
  def clone
    m = Mate.new
    @mate.keys.each do |key|
      m.set(key, @mate[key])
    end
    m
  end

  #### path? ####
  def path?(s, t)
    if @mate[s] == t && @mate[t] == s && size == 2
      true
    else
      false
    end
  end
end

########################################
# mate hash
########################################
class MateHash
  #### new ####
  def initialize
    @hash = Hash.new(nil)     # @hash[m.to_ky] = node
  end

  #### member? ####
  def member?(m)
    if @hash[m.to_key] != nil
      true
    else
      false
    end
  end

  #### size ####
  def size
    @hash.size
  end

  #### add [key (mate), val] ####
  def add(mate, val)
    @hash[mate.to_key] = val
  end

  #### get val of mate ####
  def get(mate)
    @hash[mate.to_key]
  end

  #### nodes ####
  def nodes
    @hash.values
  end

  #### clear ####
  def clear
    @hash.clear
  end
end

########################################
# ZDD node
########################################
class Node
  #### new ####
  def initialize(l, m)
    @label = l
    @mate  = m
    @count = 0
    @child = [nil, nil]
  end
  attr_reader :label, :mate, :child
  attr_accessor :count

  #### set ####
  def set_v_child(v, n)
    @child[v] = n
  end

  #### add_count ####
  def add_count(c)
    @count += c
  end

  #### show ####
  def show
    puts "----------------------------------------"
    puts "address : #{self}"
    puts "label   : #{@label}"
    puts "count   : #{@count}"
    @mate.show
    puts "0-child : #{@child[0]}"
    puts "1-child : #{@child[1]}"
    puts "----------------------------------------"
  end
end

########################################
# manager
########################################
class Manager
  #### new ####
  def initialize(ae)
    #### # of edges and array of edges ####
    @ne = ae.size
    @ae = ae.sort{|a, b| (a[0] <=> b[0]).nonzero? or a[1] <=> b[1] }

    #### array of matehash ####
    @mh = Array.new(@ne){ |i| MateHash.new }

    #### root node ####
    @root = Node.new(0, Mate.new)
    @root.count = 1
    add_node(@root)

    @one  = Node.new(0, nil)
    @zero = Node.new(1, nil)
  end

  #### add a node to MateHash ####
  def add_node(n)
    l = n.label
    m = n.mate
    @mh[l].add(m, n)
  end

  #### get v-child of n ####
  def get_v_child(n, v)
    #### opened ####
    return n.child[v] if n.child[v] != nil

    #### not opened ####
    l = n.label
    e = @ae[l]
    u = nil
    u = @ae[l+1] if l < @ne-1

    ch = @zero           # v-child (default 0)
    m  = (n.mate).clone  # mate for ch

    # add e as v
    t = m.add_edge(e, v)  # true if added

    #### last node ####
    if u == nil
      t = t && m.delete_node(e[0])  # true if e[0] is not an end
      t = t && m.path?(1, e[1])     # true if 1 - e[1] path

      # make child node
      ch = @one if t

      #### inter node #### 
    else
      if e[0] == 1 && e[0] != u[0]
        t = t && !(m.look(1) == 0 || m.look(1) == nil)  # true if node 1 is an end
      elsif e[0] != u[0] # delete if needed
        t = t && m.delete_node(e[0])                    # true if e[0] is not an end
      end

      # make child node
      if t
        if @mh[l+1].member?(m)   # existing node
          ch = @mh[l+1].get(m)
        else
          ch = Node.new(l+1, m)  # new node
          @mh[l+1].add(m, ch)
        end
      end
    end

    #### set & add to hash ####
    n.set_v_child(v, ch)

    return ch
  end

  #### open all nodes & count up####
  def solve
    # open all nodes
    for i in 0..@ne-1
      puts "depth = #{i} / #{@ne}: #{@mh[i].size} nodes"
      @mh[i].nodes.each do |n|
        open_node(n)
      end
      @mh[i].clear
    end

    # return the count of 1-terminal
    @one.count
  end

  #### open a node ####
  def open_node(n)
    for v in 0..1
      ch = get_v_child(n, v)
      ch.count += n.count
    end
  end
end

################################################################################
# main
################################################################################
g = GridGraph.new(@n)
m = Manager.new(g.edge)
puts "n = #{@n}"
puts "e = #{(g.edge).size}"
puts "#{m.solve} paths"
