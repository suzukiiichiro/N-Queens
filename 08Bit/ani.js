<html>
<body>
<script>
//https://fuuno.net/web02/arrow/arrow.html
//アロー関数
//(変数/定数名) = (引数) => { 処理 };
//const foo = a => a+2;
//const foo = function(a) {
//  return a+2;
//      };
function queen(n) {
  // a array of numerical sequence from 0 to n - 1
  var sequence = enumerateInterval(0, n - 1)
  console.log(sequence);
  // returns the sub-solution for n-queen's problem while placing k(k < n) queens on the first k rows of the board
  function queenRow(k) {
    if (k == 0) {
      return [[]]
    } else {
      var restQueens = queenRow(k - 1)
      // just place the nth queen on any place of the new row to generate a bunch of new solutions
      //まず全ての組み合わせを作る
      //0,1,2,3,4,5,6,7
      //->0,0 0,1 0,2 0,3 0,4 0,5 0,6 0,7 1,0 1,1 ....
      console.log("before");
      console.log(restQueens);
      var solutions = combine(restQueens, sequence)
      console.log("after");
      console.log(solutions);
      // and filter solutions that is safe
      return filter(function(positions){
        return safe(positions)      
      }, solutions)
    }
  }
  return queenRow(n)
}

// a1 is an array of array, a2 is an array
// this function will append every element in a2 into every element(array) in a1 and flatten the result
// ex. combine([[0, 1], [2, 3]], [4, 5]) will return [ [ 0, 1, 4 ], [ 0, 1, 5 ], [ 2, 3, 4 ], [ 2, 3, 5 ] ]
function combine(a1, a2) {
  return flatMap((x) => {
    return map((y) => {
      if (typeof x == 'object') {
        var t = x.slice()
        t.push(y)
        return t
      } else {
        return [x, y]
      }
    }, a2)
  }, a1)
}

// test if the positions of queens is not attacking each other
// because of recursion, we can assume that all but the last queen is safe.
// therefore we only check the last queen against other queens
//再帰して１行ずつ進んでいっているので最後の行についてだけ他のクイーンと抵触しないかチェックする
function safe(positions) {
  //引数なしのslice
  //引数なし arr.slice() でも呼び出すことができ、これは arr のコピーを生成します。これは、オリジナルの配列に影響を与えない形でさらに変換するためのためのコピーを取得するのによく使用されます。
  var pos = positions.slice()
  //pop() メソッドは配列の最後の要素を取り除き、呼び出し元にその値を返します
  //pop することによって i は最後の要素 pos は最後の要素以外全部
  var i = pos.pop()
  return _safe(i, pos)
}

function _safe(x, positions) {
  return !included(x, positions) && !diagonal(x, positions)
}
//downチェック
function included(x, arr) {
  //https://www.zunouissiki.com/js-array-some/
  //someの使い方
  //配列内のいずれかの要素が条件に合致しているかを判定する
  //縦列をチェックする
  //最終行以外のクイーンをforで回して最終行のクイーンの位置と同じかチェックする
  return arr.some(function(i){ 
    return i == x
  })
}
//left rightチェック
function diagonal(x, arr) {
  //some で引数が２つある場合
  //some(function(element, index) { /* … */ })
  //element 配列内で現在処理されている要素です。
  //index 現在処理されている要素の添字です。
  //x - arr.length == t - i left チェック
  //x + arr.length == t + i right チェック
  return arr.some(function(t, i){
    return x - arr.length == t - i || x + arr.length == t + i
  }
  )
  
}

function enumerateInterval(low, high) {
  var result = []
  for (var i = low; i <= high; i++) {
    result.push(i)
  }
  return result
}

function flatMap(proc, list) {
  return flatten(map(proc, list))
}

function flatten(arr) {
  return arr.reduce((a, b) => {
    return a.concat(b)
  }, [])
}

function map(proc, list) {
  return list.map(proc)
}

function filter(proc, list) {
  //items.filter( コールバック関数 )
  //「filter」の中で、特定の条件を与えて配列データを取得したい内容を「コールバック関数」に書くことで、任意のデータを抽出して新しい配列を生成します。
  //combineで作った候補１個ずつに対してsafeメソッドでdown,left,rightにひっかからないかフィルタリングしてひっかからない物だけ抽出する
  return list.filter(proc)
}

// test
var result = queen(8)
console.log(result)
console.log(result.length) // 92
</script>
</body>
</html>

