import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import category_detector as category
import vehicle_detector as vehicle

class VehicleTests(unittest.TestCase):
    def prediction(self, size=200, class_index=2, score=.9, center=(208,208)):
        values=np.zeros((3549,85),dtype=np.float32)
        index=26*52+26
        values[index,:2]=np.array(center)/8-[26,26]
        values[index,2:4]=np.log(size/8)
        values[index,4]=score
        values[index,5+class_index]=score
        return values

    def test_only_significant_central_car_is_evidence(self):
        self.assertIsNotNone(vehicle.select_car(self.prediction(),416,416,1))
        self.assertIsNone(vehicle.select_car(self.prediction(size=20),416,416,1))
        self.assertIsNone(vehicle.select_car(self.prediction(class_index=7),416,416,1))
        self.assertIsNone(vehicle.select_car(self.prediction(score=.4),416,416,1))
        self.assertIsNone(vehicle.select_car(self.prediction(center=(10,10)),416,416,1))
        bad=self.prediction();bad[0,0]=np.nan
        with self.assertRaises(ValueError):vehicle.select_car(bad,416,416,1)

    def test_corrupt_download_never_becomes_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            response=MagicMock();response.__enter__.return_value=response
            response.iter_content.return_value=[b'not a model']
            with patch.object(vehicle,'ROOT',Path(directory)),patch.object(vehicle,'_retry_after',0),patch('requests.get',return_value=response):
                with self.assertRaises(ValueError):vehicle.model_path()
                self.assertFalse((Path(directory)/'yolox_nano.onnx').exists())
                self.assertFalse((Path(directory)/'yolox_nano.download').exists())

    def test_fallback_does_not_override_property_or_run_in_property_mode(self):
        image=io.BytesIO();Image.new('RGB',(300,300),'white').save(image,format='PNG')
        session=MagicMock();session.run.return_value=[np.zeros((1,1000),dtype=np.float32)]
        labels=['unused']*1000
        evidence={'confidence':.9,'area':.5,'class':'car','box':[0,0,200,200]}
        with patch.object(category,'_session',session),patch.object(category,'_labels',labels),patch.object(vehicle,'dominant_car',return_value=evidence) as detect:
            self.assertEqual(category.classify(image.getvalue(),'car')['status'],'compatible')
            detect.reset_mock()
            self.assertEqual(category.classify(image.getvalue(),'realestate')['status'],'uncertain')
            detect.assert_not_called()
            labels[500]='studio couch'
            logits=np.zeros((1,1000),dtype=np.float32);logits[0,500]=20
            session.run.return_value=[logits]
            self.assertEqual(category.classify(image.getvalue(),'car')['status'],'mismatch')
            detect.assert_not_called()

    def test_optional_failure_preserves_uncertain_decision(self):
        image=io.BytesIO();Image.new('RGB',(300,300),'white').save(image,format='PNG')
        session=MagicMock();session.run.return_value=[np.zeros((1,1000),dtype=np.float32)]
        with patch.object(category,'_session',session),patch.object(category,'_labels',['unused']*1000),patch.object(vehicle,'dominant_car',side_effect=RuntimeError('unavailable')):
            result=category.classify(image.getvalue(),'car')
            self.assertEqual(result['status'],'uncertain')
            self.assertEqual(result['vehicle_check']['status'],'unavailable')

if __name__=='__main__':unittest.main()
