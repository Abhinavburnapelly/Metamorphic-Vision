const express = require("express");
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { parse } = require("csv-parse");

const app = express();
app.use(cors()); // Allows incoming requests from any IP

const folderPath3 = path.join(__dirname, 'test/testing');

fs.readdir(folderPath3, (err, files) => {
  if (err) {
    console.error('Error reading folder:', err);
    res.status(500).json({ error: 'Error reading folder' });
    return;
  }

  files.forEach(file => {
    const filePath = path.join(folderPath3, file);
    fs.unlinkSync(filePath); // Delete each file in the folder
  });
});

const storage2 = multer.diskStorage({
    destination: function (req, file, callback) {
        callback(null, path.join(__dirname, 'test/testing'));
    },
    filename: function (req, file, callback) {
        callback(null, file.originalname);
    }
});

const upload2 = multer({ storage: storage2 });

app.post("/ref", upload2.single("file"), (req, res) => {
        
    console.log(req.body);
    console.log(req.file);
    console.log('good boy')
    
    res.json({ message: "File(s) uploaded successfully" });
});
//-------------------------------------------------------------------------------------------------------------------

app.get("/result",(req,res)=>{

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
            if(result.length>1){res.json({ result: result[1][1] });

            }
            console.log("File read successful");
        });
    });
//-------------------------------------------------------------------------------------------------------------------
app.listen(5000, function(){
    console.log("Server running on port 5000");
});