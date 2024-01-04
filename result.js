const { parse } = require("csv-parse");

function result(){
    const fs = require("fs");

    // specify the path of the CSV file
    const path = "submission.csv";

    // Create a readstream
    // Parse options: delimiter and start from line 1
    const result=[];
    fs.createReadStream(path)
      .pipe(parse({ delimiter: ",", from_line: 1 }))
      .on("data", function (row) {
        // executed for each row of data
        console.log(row);
        result.push(row);
      })
      .on("error", function (error) {
        // Handle the errors
        console.log(error.message);
      })
      .on("end", function () {
        if(result.length>1){
            const temp=document.getElementById('result');
            temp.innerHTML=result[1][1];
        }
        console.log("File read successful");
      });
}