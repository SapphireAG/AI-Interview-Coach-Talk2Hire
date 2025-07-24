import 'package:flutter_application_2/feedback_page.dart';
import 'package:flutter_application_2/landing_page.dart';
import 'package:flutter_application_2/questions_page.dart';
// import 'package:flutter_frontend/questions_page.dart';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_application_2/login_page.dart';
import 'package:flutter_application_2/report_page.dart';
import 'questions_screen.dart';
import 'feedback_page.dart';
import 'report_page.dart';

void main() async{
  WidgetsFlutterBinding.ensureInitialized();
  await GlobalAudioScope().ensureInitialized();
  runApp(MyApp());
}  

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      
      title: 'Interview App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const LoginPage(),
        '/questions_screen': (context) => const QuestionsPage(),
        '/questions_page': (context) => const Quiz(username: '',),
        '/login_page':(context)=> const LoginPage(),
        '/landing_page':(context)=> const LandingPage(username: '',),
        '/report_page':(context)=> const ReportPage()
      },
    );
  }
}

// void main() {
//   runApp(MaterialApp(
//     debugShowCheckedModeBanner: false,
//     home: LoginPage(),
//   ));
// }

// void main() {
//   runApp(MaterialApp(
//     debugShowCheckedModeBanner: false,
//     home: QuestionsPage(),
//   ));
// }

// void main() {
//   runApp(MaterialApp(
//     debugShowCheckedModeBanner: false,
//     home: ReportPage(username: 'jahnavi'),
//   ));
// }