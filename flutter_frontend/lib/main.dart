import 'package:flutter_application_2/questions_page.dart';
// import 'package:flutter_frontend/questions_page.dart';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'login_page.dart';
import 'questions_screen.dart';

void main() async{
  WidgetsFlutterBinding.ensureInitialized();
  await GlobalAudioScope().ensureInitialized();
  runApp(Quiz());
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
