import unittest
import json
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from werkzeug.datastructures import FileStorage
import sys
from server import app, read_data, write_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestServer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        app.config['TESTING'] = True
        
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.test_data_file = os.path.join(self.test_dir, 'test_data.json')
        self.test_upload_folder = os.path.join(self.test_dir, 'uploads')
        os.makedirs(self.test_upload_folder, exist_ok=True)
        
        # Sample test data
        self.sample_data = [
            {
                "id": 1,
                "subject": "Test Subject",
                "date": "01-01-2024",
                "book": "Test Book",
                "content": "Test content",
                "number_lesson": 5,
                "duration_lesson": 2,
                "message_CLOs": []
            }
        ]
        
        # Write sample data to test file
        with open(self.test_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.sample_data, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('server.DATA_FILE')
    def test_read_data_success(self, mock_data_file):
        """Test successful data reading"""
        mock_data_file.__str__ = lambda: self.test_data_file
        mock_data_file.return_value = self.test_data_file
        
        with patch('server.DATA_FILE', self.test_data_file):
            result = read_data()
            self.assertEqual(result, self.sample_data)

    @patch('server.DATA_FILE')
    def test_write_data_success(self, mock_data_file):
        """Test successful data writing"""
        mock_data_file.__str__ = lambda: self.test_data_file
        mock_data_file.return_value = self.test_data_file
        
        new_data = [{"id": 2, "subject": "New Subject"}]
        
        with patch('server.DATA_FILE', self.test_data_file):
            write_data(new_data)
            
        # Verify data was written correctly
        with open(self.test_data_file, 'r', encoding='utf-8') as f:
            written_data = json.load(f)
            self.assertEqual(written_data, new_data)

    @patch('server.read_data')
    def test_get_courses_success(self, mock_read_data):
        """Test GET /courses endpoint"""
        mock_read_data.return_value = self.sample_data
        
        response = self.app.get('/courses')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, self.sample_data)

    @patch('server.write_data')
    def test_update_data_success(self, mock_write_data):
        """Test POST /update-data endpoint"""
        test_data = {"test": "data"}
        
        response = self.app.post('/update-data',
                                json=test_data,
                                content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'message': 'Cập nhật thành công'})
        mock_write_data.assert_called_once_with(test_data)

    def test_process_pdf_no_file(self):
        """Test POST /process-pdf with no file"""
        response = self.app.post('/process-pdf')
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Không tìm thấy file", response.json['error'])

    def test_process_pdf_empty_filename(self):
        """Test POST /process-pdf with empty filename"""
        data = {'file': (FileStorage(), '')}
        
        response = self.app.post('/process-pdf', data=data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Không có file nào được chọn", response.json['error'])

    @patch('server.process_pdf')
    @patch('server.UPLOAD_FOLDER')
    def test_process_pdf_success(self, mock_upload_folder, mock_process_pdf):
        """Test successful PDF processing"""
        mock_upload_folder.__str__ = lambda: self.test_upload_folder
        mock_upload_folder.return_value = self.test_upload_folder
        
        mock_process_pdf.return_value = {
            "book": "Test Book",
            "content": "Test Content"
        }
        
        # Create a test file
        test_file = FileStorage(
            stream=open(__file__, 'rb'),
            filename='test.pdf',
            content_type='application/pdf'
        )
        
        with patch('server.UPLOAD_FOLDER', self.test_upload_folder):
            response = self.app.post('/process-pdf',
                                   data={'file': test_file})
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['book'], "Test Book")
        self.assertEqual(response.json['content'], "Test Content")

    @patch('server.process_pdf')
    def test_process_pdf_exception(self, mock_process_pdf):
        """Test PDF processing with exception"""
        mock_process_pdf.side_effect = Exception("Processing error")
        
        test_file = FileStorage(
            stream=open(__file__, 'rb'),
            filename='test.pdf',
            content_type='application/pdf'
        )
        
        response = self.app.post('/process-pdf',
                               data={'file': test_file})
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("error", response.json)

    @patch('server.read_data')
    @patch('server.write_data')
    @patch('server.process_pdf')
    @patch('server.chat_CLO')
    @patch('server.chiasach')
    @patch('server.UPLOAD_FOLDER')
    def test_create_course_success(self, mock_upload_folder, mock_chiasach, 
                                 mock_chat_clo, mock_process_pdf, 
                                 mock_write_data, mock_read_data):
        """Test successful course creation"""
        mock_upload_folder.__str__ = lambda: self.test_upload_folder
        mock_upload_folder.return_value = self.test_upload_folder
        
        mock_read_data.return_value = self.sample_data
        mock_process_pdf.return_value = {
            "book": "New Book",
            "content": "New Content"
        }
        mock_chat_clo.return_value = "Generated CLO"
        
        test_file = FileStorage(
            stream=open(__file__, 'rb'),
            filename='test.pdf',
            content_type='application/pdf'
        )
        
        form_data = {
            'file': test_file,
            'subject': 'New Subject',
            'numberLesson': '5',
            'durationLesson': '2'
        }
        
        with patch('server.UPLOAD_FOLDER', self.test_upload_folder):
            response = self.app.post('/create-course', data=form_data)
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['id'], 2)

    @patch('server.read_data')
    @patch('server.write_data')
    @patch('server.delete_retriever')
    @patch('server.UPLOAD_FOLDER')
    def test_delete_course_success(self, mock_upload_folder, mock_delete_retriever,
                                 mock_write_data, mock_read_data):
        """Test successful course deletion"""
        mock_upload_folder.__str__ = lambda: self.test_upload_folder
        mock_upload_folder.return_value = self.test_upload_folder
        
        mock_read_data.return_value = self.sample_data
        mock_delete_retriever.return_value = True
        
        with patch('server.UPLOAD_FOLDER', self.test_upload_folder):
            response = self.app.delete('/delete-course/1')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])

    @patch('server.read_data')
    @patch('server.write_data')
    @patch('server.chat_syllabus')
    def test_initialize_syllabus_success(self, mock_chat_syllabus, 
                                       mock_write_data, mock_read_data):
        """Test successful syllabus initialization"""
        mock_read_data.return_value = self.sample_data
        mock_chat_syllabus.return_value = "Generated Syllabus"
        
        request_data = {
            'courseId': 1,
            'selectedCLO': 'Test CLO'
        }
        
        response = self.app.post('/initialize-syllabus',
                               json=request_data,
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['message'], "Generated Syllabus")

    def test_initialize_syllabus_course_not_found(self):
        """Test syllabus initialization with non-existent course"""
        with patch('server.read_data', return_value=[]):
            request_data = {
                'courseId': 999,
                'selectedCLO': 'Test CLO'
            }
            
            response = self.app.post('/initialize-syllabus',
                                   json=request_data,
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 404)
            self.assertIn("Course not found", response.json['error'])

    @patch('server.read_data')
    @patch('server.write_data')
    @patch('server.chat_CLO')
    def test_chat_clo_success(self, mock_chat_clo, mock_write_data, mock_read_data):
        """Test successful CLO chat"""
        mock_read_data.return_value = self.sample_data
        mock_chat_clo.return_value = "Chat response"
        
        request_data = {
            'courseId': 1,
            'message': 'Test message',
            'chatType': 'CLOs'
        }
        
        response = self.app.post('/chat',
                               json=request_data,
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['message'], "Chat response")

    def test_chat_invalid_type(self):
        """Test chat with invalid chat type"""
        with patch('server.read_data', return_value=self.sample_data):
            request_data = {
                'courseId': 1,
                'message': 'Test message',
                'chatType': 'InvalidType'
            }
            
            response = self.app.post('/chat',
                                   json=request_data,
                                   content_type='application/json')
            
            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid chat type", response.json['error'])

    @patch('server.read_data')
    @patch('server.write_data')
    @patch('server.chiabuoi')
    @patch('server.UPLOAD_FOLDER')
    def test_split_sessions_success(self, mock_upload_folder, mock_chiabuoi,
                                  mock_write_data, mock_read_data):
        """Test successful session splitting"""
        mock_upload_folder.__str__ = lambda: self.test_upload_folder
        mock_upload_folder.return_value = self.test_upload_folder
        
        mock_read_data.return_value = self.sample_data
        
        request_data = {
            'courseId': 1,
            'message': 'Test syllabus content'
        }
        
        with patch('server.UPLOAD_FOLDER', self.test_upload_folder):
            response = self.app.post('/split-sessions',
                                   json=request_data,
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])

    @patch('server.read_data')
    @patch('server.write_data')
    @patch('server.get_sumarize')
    @patch('server.save_retriever')
    @patch('server.generate_lesson_plan')
    @patch('server.UPLOAD_FOLDER')
    def test_initialize_lesson_success(self, mock_upload_folder, mock_generate_plan,
                                     mock_save_retriever, mock_get_sumarize,
                                     mock_write_data, mock_read_data):
        """Test successful lesson initialization"""
        mock_upload_folder.__str__ = lambda: self.test_upload_folder
        mock_upload_folder.return_value = self.test_upload_folder
        
        mock_read_data.return_value = self.sample_data
        mock_retriever = MagicMock()
        mock_retriever.vectorstore = MagicMock()
        mock_retriever.docstore = MagicMock()
        mock_get_sumarize.return_value = (mock_retriever, "Text summaries")
        mock_save_retriever.return_value = True
        mock_generate_plan.return_value = "Generated lesson plan"
        
        # Create test PDF file
        test_pdf_path = os.path.join(self.test_upload_folder, '1', 'chiabuoi', 'Buoi_1.pdf')
        os.makedirs(os.path.dirname(test_pdf_path), exist_ok=True)
        with open(test_pdf_path, 'w') as f:
            f.write('test')
        
        request_data = {
            'courseId': 1,
            'lessonNumber': 1
        }
        
        with patch('server.UPLOAD_FOLDER', self.test_upload_folder):
            response = self.app.post('/initialize-lesson',
                                   json=request_data,
                                   content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['message'], "Generated lesson plan")

    def test_initialize_lesson_pdf_not_found(self):
        """Test lesson initialization with missing PDF"""
        with patch('server.read_data', return_value=self.sample_data):
            with patch('server.UPLOAD_FOLDER', self.test_upload_folder):
                request_data = {
                    'courseId': 1,
                    'lessonNumber': 1
                }
                
                response = self.app.post('/initialize-lesson',
                                       json=request_data,
                                       content_type='application/json')
                
                self.assertEqual(response.status_code, 404)
                self.assertIn("PDF for lesson 1 not found", response.json['error'])

    @patch('server.read_data')
    @patch('server.write_data')
    @patch('server.load_retriever')
    @patch('server.chat_lesson')
    def test_chat_lesson_success(self, mock_chat_lesson, mock_load_retriever,
                                mock_write_data, mock_read_data):
        """Test successful lesson chat"""
        mock_read_data.return_value = self.sample_data
        mock_retriever = MagicMock()
        mock_load_retriever.return_value = (mock_retriever, None)
        mock_chat_lesson.return_value = "Lesson chat response"
        
        request_data = {
            'courseId': 1,
            'lessonNumber': 1,
            'message': 'Test question'
        }
        
        response = self.app.post('/chat-lesson',
                               json=request_data,
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['message'], "Lesson chat response")

    @patch('server.list_retrievers')
    def test_list_retrievers_success(self, mock_list_retrievers):
        """Test successful retriever listing"""
        mock_list_retrievers.return_value = ['retriever1', 'retriever2']
        
        response = self.app.get('/list-retrievers')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['retrievers'], ['retriever1', 'retriever2'])

    @patch('server.delete_retriever')
    def test_delete_retriever_success(self, mock_delete_retriever):
        """Test successful retriever deletion"""
        mock_delete_retriever.return_value = True
        
        request_data = {
            'courseId': 1,
            'lessonNumber': 1
        }
        
        response = self.app.post('/delete-retriever',
                               json=request_data,
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])

if __name__ == '__main__':
    unittest.main()