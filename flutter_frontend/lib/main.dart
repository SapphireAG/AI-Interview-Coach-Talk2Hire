import 'package:flutter_application_2/questions_page.dart';
// import 'package:flutter_frontend/questions_page.dart';
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';

void main() async{
  WidgetsFlutterBinding.ensureInitialized();
  await GlobalAudioScope().ensureInitialized();
  runApp(Quiz());
}