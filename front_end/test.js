function decode(idx) {
    let chars = [];
    for(id of idx) {
        chars.push(id + 32);
    }
    return String.fromCharCode(...chars);
}

console.log(decode([1, 2, 0, 3]));