from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <br>
    <div>
        <h1>Welcome to my Portfolio</h1>
    </div>
    <div>
        <ul>
            <li><a href="#">Home</a></li>
            <li><a href="#">About</a></li>
            <li><a href="#">Education</a></li>
            <li><a href="#">Projects</a></li>
            <li><a href="#">Contact</a></li>


        </ul>
    </div>
    <br>
    <h1>About</h1>
    <table  border =10  cellspacing="5" >
        <tr align="center">
            <td>
                <h1>Mayank Rauthan</h1>
            </td>
            <td rowspan="2">
                <img src="pic.jpg" width="200" >
            </td>
            <tr align="center">
                <td><p>I am <b>Mayank Rauthan</b>. I am in my 6th semester pusuing computer
                    science engineering from <b>Graphic Era Hill University</b>. I am very
                    passionate about programming, problem solving and learning 
                    various new tehnology
               </p>
            </td>
            </tr>
        </tr>
    </table>
    <br>
    <h1>
        Education
    </h1>
    <table border="5" cellspacing="2" cellpadding="10">
        <tr align="center">
            <th>Education</th>
            <th colspan="2">Institute</th>
            <th>Year</th>
            <th>Score</th>
        </tr>
        <tr align="center">
            <td>Computer Science Engineering</td>
            <td colspan="2">Graphic Era Hill University</td>
            <td>2021-2025</td>
            <td>8.3 CGPA</td>
        </tr>
        <tr align="center">
            <td>Intermediate</td>
            <td >B.R.M.S</td>
            <td>CBSE</td>
            <td>2019-2021</td>
            <td>95.2%</td>
        </tr>
        <tr align="center">
            <td>Matriculation</td>
            <td >St.Thomas, pauri</td>
            <td>ICSE</td>
            <td>2019</td>
            <td>90%</td>
        </tr>^
    </table>
    <br>
    <h1>Skills</h1>
    <ul>
        <li><b> Languages: </b>Java, Python, C, C++, Javascript, CSS, SQL, XML, HTML
        </li>
        <br>
        <li><b>Tools and Frameworks: </b>Android Studio, GitHub, Pandas, Sklearn, Jupyter
        </li><br>
        <li><b>Platforms/Database: </b> LINUX, Google FireBase, Oracle, MySQL</li>
        <br>
        <li><b> Soft Skills: </b>Problem-Solving, Team Spirit, Pressure Handling</li>
    </ul>
    <br>
    <h1>Academic Projects</h1>
    <ul>
        <li><b>Student Management App</b>
        <ul>
            <li> A cloud-based student management android app designed to maintain
                student records, track fee payments, manage student enrollment and provide
                financial performance of the institution.</li>
            <li>
                Technology Used : Java, Android Studio, Retrofit , XML, Firebase
            </li>
        </ul>
        <br>
        </li>
        <li><b>Newshub - API based News App</b>
        <ul>
            <li>
                Developed an Android app that provides news articles, allows article search, and
                bookmarking for later reading
            </li>
            <li>
                Technology Used : Java, Android Studio, Retrofit , XML, Rest API

            </li>
        </ul>
         </li>
         <br>

         <li><b> Notes - DBMS based Note App</b>
            <ul>
                <li>
                    Developed a DBMS-based note management application with Google Sign-In,
                    enabling individualized note CRUD operations for each user
                </li>
                <li>
                    Technology Used : Java, Android Studio, Sqlite, Google Sign-in

                </li>
            </ul>
             </li>
    </ul>
    <br>

    <h1>Achievements</h1>
    <ul > 
        <li><b>President of Pragyan Club</b>
            <ul>Regularly conducted workshops and events in the domain of Soft Skills, with each
                event drawing participation from over 150 college students</ul>
        </li>
        <br>
        <li><b> Part-Time Programming Tutor| Infinity Institute</b>
            <ul>Provided tutoring in Java and Python to students of various age groups and skill
                levels, leading to students achieving high marks in their assignments and exams.
                </ul>
        </li>
    </ul>
    <br>
<footer>
    <h1>Contact</h1>
    <p><b>Email: </b>mayankrauthan02@gmail.com</p>
    <p><b>Linkedln: </b> <a href=https://www.linkedin.com/in/mayank-rauthan-82b385226>\MayankRauthan</ahref></a></p>
    <p><b>Mobile no: </b>7617671745</p>
</footer>

  
</body>
</html>
    """
