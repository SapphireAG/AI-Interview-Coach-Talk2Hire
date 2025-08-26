import 'package:flutter/material.dart';
import 'package:flutter_application_2/landing_page.dart';
import 'package:flutter_application_2/questions_page.dart';
import 'package:flutter_application_2/questions_screen.dart';
import 'package:http/http.dart' as http;

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});
  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  void _getLoginInfo() {
    String username = _usernameController.text; 
    print("Username: $username");
    String password = _passwordController.text;
    print("Password: $password");
  }

Future<void> sendUsername(String username) async {
  final uri = Uri.parse("http://127.0.0.1:8000/username/");
  final response = await http.post(
    uri,
    body: {'username': username},
  );

  if (response.statusCode == 200) {
   
    print("Username sent successfully");
  } else {
    print("Username send failed");
  }
}


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color.fromARGB(255, 249, 250, 251),
      // appBar: AppBar(
      //   toolbarHeight: 100.0,
      //   backgroundColor: Color.fromARGB(255, 91, 106, 234),
      //   title: Center(
      //     child: ClipRRect(
      //       borderRadius: BorderRadius.circular(
      //         12.0,
      //       ), // Adjust the radius as needed
      //       child: Image.asset(
      //         'assets/logo.jpg',
      //         height: 50.0,
      //         width: 200.0,
      //         scale: 1.0,
      //         fit: BoxFit.cover,
      //       ),
      //     ),
      //   ),
      // ), 
      
      body: Center(
        child: Column(
          
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Center(
              child: Image.asset(
                'assets/NEW_logo.png',
                height: 250 ,
              ),
            ),
             
            Container(
              height: 300.0,
              width: 300.0,
              padding: EdgeInsets.all(20.0),
              decoration: BoxDecoration(
                color: Color.fromARGB(255, 237, 237, 237),
                borderRadius: BorderRadius.circular(20.0),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black12,
                    blurRadius: 4,
                    offset: Offset(2, 2),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment:
                    CrossAxisAlignment.start, // Aligns elements to the left
                children: [
                  Row(
                    children: [
                      Icon(
                        
                        Icons.account_circle, 
                        size: 40,
                        color:Colors.black54,
                      ),
                      SizedBox(width: 10),
                      Text(
                        textAlign: TextAlign.center,
                        "Login To Talk2Hire",
                        style: TextStyle(
                          color: Color.fromARGB(255, 6, 178, 200),
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 20),
                  // Username label and field
                  Text(
                    'Username',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.black54,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 5.0), //
                  Padding(
                    padding: EdgeInsets.only(top: 5.0),
                    child: TextField(
                      controller: _usernameController,
                      decoration: InputDecoration(
                        fillColor: Colors.white,
                        filled: true,
                        hintText: 'Enter Username',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(10.0),
                        ),
                      ),
                    ),
                  ),

                  SizedBox(height: 20),

                  // Password label and field
                  Text(
                    'Password',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.black54,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 2.0),
                  Padding(
                    padding: EdgeInsets.only(top: 5.0),
                    child: TextField(
                      controller: _passwordController,
                      obscureText: true,
                      decoration: InputDecoration(
                        fillColor: Colors.white,
                        filled: true,
                        hintText: 'Enter Password',
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(10.0),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: ()  {
                _getLoginInfo();
                sendUsername(_usernameController.text); // Send username to backend
                Navigator.push(
                context,
                MaterialPageRoute(
                builder: (context) => LandingPage(username: _usernameController.text),
                
                  ),
                );

              
                // Navigator.pushNamed(context, '/questions_screen');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Color.fromARGB(255, 190, 233, 236),
                padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              ),
              
              child: Text(
                'Continue',
                style: TextStyle(
                  fontSize: 22,
                  color: Colors.black,
                  fontStyle: FontStyle.normal,
                ),
              ),
            ),
            SizedBox(height: 50),
          ],
        ),
      ),
    );
  }
}