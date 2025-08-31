import 'package:flutter_application_2/feedback_page.dart';
import 'package:flutter_application_2/landing_page.dart';
import 'package:flutter_application_2/questions_page.dart';
// import 'package:flutter_frontend/questions_page.dart';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:flutter_application_2/login_page.dart';
import 'package:flutter_application_2/report_page.dart';
import 'questions_screen.dart';

import 'report_page.dart';

// void main() async{
//   WidgetsFlutterBinding.ensureInitialized();
//   await GlobalAudioScope().ensureInitialized();
//   runApp(MyApp());
// }  

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
      
//       title: 'Interview App',
//       debugShowCheckedModeBanner: false,
//       theme: ThemeData(
//         primarySwatch: Colors.blue,
//       ),
//       initialRoute: '/',
//       routes: {
//         '/': (context) => const LoginPage(),
//         '/questions_screen': (context) => const QuestionsPage(),
//         '/questions_page': (context) => const Quiz(username: '',),
//         '/login_page':(context)=> const LoginPage(),
//         '/landing_page':(context)=> const LandingPage(username: '',),
//         '/report_page':(context)=> const ReportPage()
//       },
//     );
//   }
// } 

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
//     routes: {
//   '/questions_page': (context) => Quiz(username: '',), // or your actual widget
// }
//   ));
// }



// void main() {
//   runApp(MaterialApp(
//     debugShowCheckedModeBanner: false,
//     home: ReportPage(username: 'jahnavi'),
//   ));
// }



void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // await GlobalAudioScope().ensureInitialized(); // This line may cause issues, ensure the package is correctly configured
  runApp(const MyApp());
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
      // Use onGenerateRoute for dynamic routes
      onGenerateRoute: (settings) {
        switch (settings.name) {
          case '/':
            return MaterialPageRoute(builder: (_) => const LoginPage());
          case '/login_page':
            return MaterialPageRoute(builder: (_) => const LoginPage());
          case '/landing_page':
            final username = settings.arguments as String;
            return MaterialPageRoute(
              builder: (_) => LandingPage(username: username),
            );
          case '/questions_screen':
             final username = settings.arguments as String;
             return MaterialPageRoute(
                // Pass username to QuestionsPage
               builder: (_) => QuestionsPage(username: username),
             );
          case '/questions_page':
            final args = settings.arguments as Map<String, dynamic>;
            final username = args['username'] as String;
            final questions = args['questions'] as List<dynamic>;
            return MaterialPageRoute(
              // Pass username via constructor and questions via settings
              builder: (_) => Quiz(username: username),
              settings: RouteSettings(arguments: questions),
            );
          case '/report_page':
            return MaterialPageRoute(builder: (_) => const ReportPage());
          default:
            return MaterialPageRoute(builder: (_) => const LoginPage());
        }
      },
      // You can define a custom 404/error page if a route is not found
      onUnknownRoute: (settings) {
        return MaterialPageRoute(builder: (_) => const LoginPage());
      },
    );
  }
}